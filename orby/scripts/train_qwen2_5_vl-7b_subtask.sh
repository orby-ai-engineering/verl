set -x
HYDRA_FULL_ERROR=1
ENGINE=${1:-vllm}
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

export EXPERIMENT_NAME="hsmv2_distill_grpo_9k_qwen7b_bbox_batch_size_32"
export NUM_NODES=1
export TRAIN_BATCH_SIZE=32
export MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
export PROJECT_NAME=hsmv2_distilled_data_bbox
export S3_CHECKPOINT_DIR=s3://orby-osu-va/subtask/verl/hsmv2_distill/checkpoints/$EXPERIMENT_NAME

# Set training environment variables
export TRAIN_FILES=/workspace/datasets/subtask/executor_dataset/train_verl_data.parquet
export VAL_FILES=/workspace/datasets/subtask/executor_dataset/test_verl_data.parquet
export REWARD_FN=training_reward_func
export REWARD_FILE=orby/reward/subtask.py
export COORDINATES_METRIC="bbox"
export COORDINATES_GAUSSIAN_SIGMA=5
export COORDINATES_PIXEL_SQUARE_SIZE=10

echo "If you encounter OOM, try tweaking the following parameters:"
echo "data.train_batch_size"
echo "actor_rollout_ref.actor.ppo_mini_batch_size"
echo "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"
echo "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"
echo "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu"
echo "actor_rollout_ref.rollout.n"

python3 -m verl.trainer.main_ppo \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN \
    +custom_reward_function.reward_kwargs.coordinates_metric=$COORDINATES_METRIC \
    +custom_reward_function.reward_kwargs.coordinates_gaussian_sigma=$COORDINATES_GAUSSIAN_SIGMA \
    +custom_reward_function.reward_kwargs.coordinates_pixel_square_size=$COORDINATES_PIXEL_SQUARE_SIZE \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILES \
    data.val_files=$VAL_FILES \
    data.train_batch_size=32 \
    data.max_prompt_length=7680 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.max_model_len=8192 \
    +actor_rollout_ref.rollout.limit_images=3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=100 \
    trainer.total_epochs=1 $@
