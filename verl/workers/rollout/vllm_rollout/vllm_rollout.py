# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import getpass
import logging
import os
from dataclasses import asdict
from types import MethodType
from typing import Any, Generator

import cloudpickle as pickle
import ray
import torch
import torch.distributed
import zmq
import zmq.asyncio
from filelock import FileLock
from torch.distributed.device_mesh import DeviceMesh
from vllm.config import LoRAConfig

from verl.utils.ray_utils import get_event_loop

import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Tuple, Generator, Dict, Any
from omegaconf import OmegaConf

try:
    from vllm.worker.worker_base import WorkerWrapperBase
except ModuleNotFoundError:
    # https://github.com/vllm-project/vllm/commit/6a113d9aed8221a9c234535958e70e34ab6cac5b
    from vllm.v1.worker.worker_base import WorkerWrapperBase

from packaging import version as vs

from verl import DataProto
from verl.third_party.vllm import VLLM_SLEEP_LEVEL, get_version
from verl.utils.device import is_npu_available
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.ray_utils import ray_noset_visible_devices
from verl.utils.vllm import TensorLoRARequest, VLLMHijack, is_version_ge
from verl.utils.vllm.vllm_fp8_utils import apply_vllm_fp8_patches, is_fp8_model, load_quanted_weights
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.utils import get_free_port, is_valid_ipv6_address
from verl.workers.rollout.vllm_rollout.utils import (
    VLLM_LORA_INT_ID,
    VLLM_LORA_NAME,
    VLLM_LORA_PATH,
    get_vllm_max_lora_rank,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


if is_version_ge(pkg="vllm", minver="0.7.3"):
    VLLMHijack.hijack()


def _check_vllm_version_for_sleep_level():
    # https://github.com/vllm-project/vllm/issues/25171
    minver = "0.11.0"
    current_version = get_version("vllm")
    if not current_version:
        logger.warning("Could not determine vLLM version, assuming an older version for sleep_level configuration.")
        return False
    return vs.parse(current_version) >= vs.parse(minver)


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        logits = original_compute_logits(*args, **kwargs)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout(BaseRollout):
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase, which is engine in single worker process."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        device_mesh: DeviceMesh,
    ):
        super().__init__(config, model_config, device_mesh)
        self.tokenizer = self.model_config.tokenizer
        self.inference_engine: WorkerWrapperBase = None
        self.address = self._init_zeromq()
        self.lora_config = (
            {"max_loras": 1, "max_lora_rank": get_vllm_max_lora_rank(self.model_config.lora_rank)}
            if self.model_config.lora_rank > 0
            else {}
        )

        if config.layered_summon or (config.expert_parallel_size > 1 and not _check_vllm_version_for_sleep_level()):
            logger.warning("Setting the sleep level to 1 may cause a memory overflow.")
            self.sleep_level = 1
        else:
            self.sleep_level = VLLM_SLEEP_LEVEL

        self.model_original_path = model_config.path
        self.model_local_path = model_config.local_path
        self.ms_modelslim_path = os.environ.get("MS_MODELSLIM_PATH", "msmodelslim")
        self._quant_cache = {}  # 量化缓存
        self._last_quant_path = None

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock(f"/tmp/verl_vllm_zmq_{getpass.getuser()}.lock"):
            context = zmq.asyncio.Context()
            self.socket = context.socket(zmq.REP)
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}_{getpass.getuser()}.ipc"
            else:
                ip = ray.util.get_node_ip_address().strip("[]")
                port, sock = get_free_port(ip)
                if is_valid_ipv6_address(ip):
                    address = f"tcp://[{ip}]:{port}"
                    self.socket.setsockopt(zmq.IPV6, 1)
                else:
                    address = f"tcp://{ip}:{port}"
            self.socket.bind(address)

        loop = get_event_loop()
        self.zmq_loop_task = loop.create_task(self._loop_forever())

        return address

    async def _loop_forever(self):
        while True:
            try:
                message = await self.socket.recv()
                method, args, kwargs = pickle.loads(message)
                result = await self._execute_method(method, *args, **kwargs)
                await self.socket.send(pickle.dumps(result))
            except Exception as e:
                logger.exception(f"vLLMAsyncRollout _loop_forever error: {e}")
                await self.socket.send(pickle.dumps(e))
                break

    def _init_worker(self, all_kwargs: list[dict[str, Any]]):
        """Initialize worker engine."""
        if not torch.distributed.is_initialized():
            initialize_global_process_group_ray()
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        device_name = "NPU" if is_npu_available else "GPU"
        all_kwargs[0]["local_rank"] = (
            0
            if not ray_noset_visible_devices()
            else int(ray.get_runtime_context().get_accelerator_ids()[device_name][0])
        )
        self.vllm_config = all_kwargs[0]["vllm_config"]
        if self.lora_config:
            lora_dtype = getattr(torch, self.config.dtype)
            self.vllm_config.lora_config = LoRAConfig(lora_dtype=lora_dtype, **self.lora_config)
        if self.config.quantization is not None:
            _SUPPORTED_QUANTIZATION = ["fp8", "torchao"]
            if self.config.quantization not in _SUPPORTED_QUANTIZATION:
                raise ValueError(
                    f"Currently only support {_SUPPORTED_QUANTIZATION} quantization, got: {self.config.quantization}"
                )

            if self.config.quantization == "fp8":
                # Apply vllm fp8 patches
                # Will remove the patch after vllm support on-the-fly quant for rollout natively.
                apply_vllm_fp8_patches()

        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def _load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)
        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    async def _execute_method(self, method: str | bytes, *args, **kwargs):
        if method == "init_worker":
            return self._init_worker(*args, **kwargs)
        elif method == "load_model":
            return self._load_model(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)

    async def resume(self, tags: list[str]):
        """Resume rollout weights or kv cache in GPU memory.

        Args:
            tags: weights or kv_cache.
        """
        if self.config.free_cache_engine:
            self.inference_engine.wake_up(tags=tags)

    async def release(self):
        """Release weights and kv cache in GPU memory."""
        if self.config.free_cache_engine:
            self.inference_engine.sleep(level=self.sleep_level)

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):
        """Update the weights of the rollout model.

        Args:
            weights: A generator that yields the name of the weight tensor and the tensor itself.
        """
        peft_config, base_sync_done = kwargs.get("peft_config", None), kwargs.get("base_sync_done", False)
        
        # mxfp8_config = getattr(self.config, 'w8a8_quantization', None)
        # # mxfp8_config = getattr(self.config, 'mxfp8_quantization', None)
        # is_mxfp8_enabled = mxfp8_config and mxfp8_config.enabled and not peft_config

        mxfp8_config = self.config.get('w8a8_quantization', {})  # 获取字典
    
        # 安全访问 enabled 属性（处理 dict 和对象两种情况）
        is_mxfp8_enabled = False
        if mxfp8_config:
            if isinstance(mxfp8_config, dict):
                is_mxfp8_enabled = mxfp8_config.get('enabled', False)
            else:  # 如果是对象
                is_mxfp8_enabled = getattr(mxfp8_config, 'enabled', False)
        
        # 检查是否是基础模型同步（不是 LoRA）
        is_mxfp8_enabled = is_mxfp8_enabled and not peft_config
        
        # if is_mxfp8_enabled:
        #     logger.info(f"MXFP8 quantization enabled, applying to base model weights")
        #     # 将权重生成器包装为量化生成器
        #     weights = self._quantize_weights_mxfp8(weights, mxfp8_config)

        if is_mxfp8_enabled:
            logger.info("W8A8 quantization enabled via msit, processing weights...")

            # 1. 将权重保存为临时 checkpoint（msmodelslim 需要路径输入）
            with tempfile.TemporaryDirectory(prefix="verl_w8a8_") as tmpdir:
                weights_dict = dict(weights)  # 收集权重
                checkpoint_path = Path(tmpdir) / "model_before_w8a8"
                print(f"=================checkpoint_path:{checkpoint_path}================")
                
                # 保存为 HF 格式（msmodelslim 支持）
                self._save_weights_as_hf_checkpoint(weights_dict, checkpoint_path)
                
                # 2. 调用 msit 量化
                quantized_path = Path(tmpdir) / "model_w8a8"
                print(f"=================quantized_path:{quantized_path}================")
                self._run_msit_quantization(
                    input_path=checkpoint_path,
                    output_path=quantized_path,
                    config=mxfp8_config
                )
                
                
                # 3. 加载量化后的权重
                weights = self._load_quantized_weights(quantized_path)

        if peft_config and base_sync_done:
            # In async mode, make sure the old lora is removed before adding the new one
            self.inference_engine.worker.remove_lora(VLLM_LORA_INT_ID)
            weights = dict(weights)
            lora_request = TensorLoRARequest(
                lora_name=VLLM_LORA_NAME,
                lora_int_id=VLLM_LORA_INT_ID,
                lora_path=VLLM_LORA_PATH,
                peft_config=asdict(peft_config),
                lora_tensors=weights,
            )
            self.inference_engine.worker.add_lora(lora_request)
            logger.info(f"vLLM load weights, loaded_params: {len(weights)}")
        else:
            from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

            model_runner = self.inference_engine.worker.model_runner
            model = model_runner.model
            patch_vllm_moe_model_weight_loader(model)

            # Add the FP8 related logic here as sharding manager has been deprecated.
            # Check if FP8 quantization is enabled and apply appropriate weight loading

            if is_mxfp8_enabled:
                logger.info("Loading MXFP8 quantized weights directly")
                model.load_weights(weights)
            elif is_fp8_model(model_runner.vllm_config):
                logger.info(f"FP8 model detected (async): {model_runner.vllm_config.quant_config}")
                # Convert bf16 weights to fp8 format before loading
                loaded_params = load_quanted_weights(weights, model_runner)
                logger.info(f"FP8 weights loaded (async), loaded_params: {len(loaded_params)}")
            else:
                logger.info("Loading standard weights (non-FP8, async)")
                model.load_weights(weights)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        """Batch generate sequences in sync mode.

        Note: vLLMAsyncRollout uses async server mode and does not support synchronous
        generation. Since SPMD mode was retired (PR #4411), the generation workflow
        should use the async server interface instead.

        Raises:
            NotImplementedError: Always raised as sync generation is not supported.
        """
        raise NotImplementedError(
            "vLLMAsyncRollout does not support synchronous generate_sequences(). "
            "The vLLM SPMD mode was retired in PR #4411. For batch generation, "
            "please use the async server interface via vLLMReplica and AsyncLLMServerManager, "
            "or use HFRollout for synchronous generation. "
            "See https://github.com/volcengine/verl/issues/4682 for more details."
        )

    # ==================== server mode public methods ====================

    def get_zeromq_address(self):
        return self.address


    def _save_weights_as_hf_checkpoint(self, weights_dict: Dict[str, torch.Tensor], save_path: Path):
        """
        将权重字典保存为 HuggingFace checkpoint 格式
        """
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 分离权重和 scale（如果已量化）
        state_dict = {}
        for name, tensor in weights_dict.items():
            # 跳过 scale（msmodelslim 会从配置文件读取）
            if name.endswith("_scale"):
                continue
            state_dict[name] = tensor
        
        # 保存权重
        torch.save(state_dict, save_path / "pytorch_model.bin")
        
        # 复制配置文件（从本地路径）
        if self.model_local_path:
            config_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
            for cfg_file in config_files:
                src = Path(self.model_local_path) / cfg_file
                if src.exists():
                    shutil.copy2(src, save_path / cfg_file)

    def _run_msit_quantization(self, input_path: Path, output_path: Path, config: Any):
        """
        调用 msit 命令行进行 W8A8 量化
        
        Args:
            input_path: 输入模型路径（HuggingFace 格式）
            output_path: 量化后模型保存路径
            config: 量化配置对象
        """
        # 构建 msit 命令
        cmd = [
            self.ms_modelslim_path,  # 如 "python -m msit" 或 "msmodelslim"
            "quant",
            "--model_path", str(input_path),
            "--save_path", str(output_path),
            "--device", "npu",  # 或 "cpu", "gpu"
            "--model_type", self.model_config.architectures[0],  # 从配置读取，如 "Qwen2ForCausalLM"
            "--quant_type", "w8a8",
            "--trust_remote_code", "True",
        ]
        
        # 可选参数
        if hasattr(config, 'calibration_samples') and config.calibration_samples > 0:
            cmd.extend(["--calibration_samples", str(config.calibration_samples)])
        
        # 执行量化
        logger.info(f"Running msit quantization: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=600  # 10分钟超时
            )
            logger.info(f"msit quantization succeeded: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"msit quantization failed: {e.stderr}")
            raise RuntimeError(f"W8A8 quantization failed: {e.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("W8A8 quantization timed out after 10 minutes")

    def _load_quantized_weights(self, quantized_path: Path) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """
        加载量化后的权重文件（mt-safe-tensors 格式）
        
        Returns:
            权重生成器，符合 vLLM load_weights 接口
        """
        # msit 可能生成 .safetensors 或 .bin
        weight_files = list(quantized_path.glob("*.safetensors")) or list(quantized_path.glob("pytorch_model.bin"))
        
        if not weight_files:
            raise FileNotFoundError(f"No weight files found in {quantized_path}")
        
        # 加载权重
        state_dict = {}
        for wf in weight_files:
            if wf.suffix == ".safetensors":
                from safetensors import safe_open
                with safe_open(wf, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        state_dict[key] = f.get_tensor(key)
            else:
                state_dict.update(torch.load(wf, map_location="cpu"))
        
        # 生成器形式返回
        for name, tensor in state_dict.items():
            yield name, tensor


    def _quantize_weights_mxfp8(self, weights: Generator[Tuple[str, torch.Tensor], None, None], config) -> Generator[Tuple[str, torch.Tensor], None, None]:
        """
        动态将权重转换为 MXFP8 格式（per-channel 量化）
        
        Args:
            weights: 原始权重生成器，产生 (name, tensor) 对
            config: MXFP8 量化配置对象
        
        Returns:
            量化后的权重生成器，产生 (name, q_tensor) 和 (name_scale, scale)
        """
        ignore_modules = getattr(config, 'ignore_modules', ["lm_head", "embed_tokens"])
        quantize_dtype = torch.float8_e4m3fn  # MXFP8 使用 E4M3 格式
        
        for name, tensor in weights:
            # 判断是否需要量化：跳过指定模块和非 2D 权重
            if any(ignore in name for ignore in ignore_modules) or tensor.dim() != 2:
                yield name, tensor
                continue
            
            # Per-channel 量化：每个输出通道一个缩放因子
            # FP8 E4M3 范围: -448 到 448
            fp8_max = 448.0
            scale = tensor.abs().max(dim=0, keepdim=True)[0] / fp8_max
            scale = scale.clamp(min=1e-12)  # 防止除零
            
            # 执行量化（无梯度跟踪）
            with torch.no_grad():
                q_tensor = (tensor / scale).clamp(-fp8_max, fp8_max).to(quantize_dtype)
                scale_fp32 = scale.squeeze(0).to(torch.float32)
            
            # 返回量化权重和缩放因子（vLLM 期望的格式）
            yield name, q_tensor
            yield f"{name}_scale", scale_fp32
            
            # 可选：清理缓存以节省显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()