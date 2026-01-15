set -x

export VLLM_ASCEND_ENABLE_NZ=0
export TORCHDYNAMO_VERBOSE=1
export HYDRA_FULL_ERROR=1
export ASCEND_RT_VISIBLE_DEVICES=5
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VERL_FILE_LOGGER_ROOT="/home/f00939291/verl/verl/outputs2"
# export HF_HUB_OFFLINE=true
#import torch
#import torch_npu
#from torch_npu.npu import amp # 导入AMP模块
#from torch_npu.contrib import transfer_to_npu # 使能自动迁移
#option = {"ACL_PRECISION_MODE":"must_keep_origin_dtype"}
#torch_npu.npu.set_option(option)


# export MS_MODELSLIM_PATH="msmodelslim"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/f00939291/dataset/train.parquet \
    data.val_files=/home/f00939291/dataset/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=90 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/home/f00939291/dataset/Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.grad_clip=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref.rollout.w8a8_quantization.enabled=True \
    +actor_rollout_ref.rollout.w8a8_quantization.msit_path="msmodelslim" \
    +actor_rollout_ref.rollout.w8a8_quantization.calibration_samples=128 \
    +actor_rollout_ref.rollout.w8a8_quantization.ignore_modules="['lm_head','embed_tokens']" \
    +actor_rollout_ref.rollout.w8a8_quantization.smoothing_strength=0.8 \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='test' \
    trainer.experiment_name='fww_05B' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=2 \
    trainer.device=npu $@
    
