# Set these variables before running the script.
export EXPERIMENT_NAME=ui-tars-subtask-sft-then-grpo
export NUM_NODES=1
export TRAIN_BATCH_SIZE=32
export MODEL_NAME=/workspace/model
export PROJECT_NAME=verl_grpo_example_subtask
export S3_CHECKPOINT_DIR=s3://orby-osu-va/verl-checkpoints/subtask/$EXPERIMENT_NAME

# Set training environment variables
export TRAIN_FILES=$HOME/data/subtask_direct_distill/mix/train/combined.parquet
export VAL_FILES=$HOME/data/subtask_direct_distill/mix/test/combined.parquet
export REWARD_FN=training_reward_func
export REWARD_FILE=orby/reward/subtask.py
export COORDINATES_METRIC="gaussian"
export COORDINATES_GAUSSIAN_SIGMA=5
export COORDINATES_PIXEL_SQUARE_SIZE=10

# Init environment
# Set environment variables
export HYDRA_FULL_ERROR=1
INTERFACE=$(ip route | grep default | awk '{print $5}' | head -1)
echo "Using interface: $INTERFACE"
export GLOO_SOCKET_IFNAME=$INTERFACE
export HF_HUB_ENABLE_HF_TRANSFER=1

# Install verl lib: https://verl.readthedocs.io/en/latest/start/install.html

# Download merged datasets
#mkdir -p ~/data/subtask_direct_distill/mix/train/
#mkdir -p ~/data/subtask_direct_distill/mix/test/
#aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/executor_reward_model_combined.parquet $VAL_FILES
#aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/executor_reward_model_combined_block512mb/part-00000-tid-3008114883591573361-eb7554a3-5626-4452-8b82-4ba9fa62e452-352-1-c000.snappy.parquet $TRAIN_FILES

# Start ray cluster and wait for all nodes

ray status
ray job submit --address="http://127.0.0.1:8265" \
	--runtime-env=verl/trainer/runtime_env.yaml \
	--no-wait \
	-- \
	python3 -m verl.trainer.main_ppo \
	custom_reward_function.path=$REWARD_FILE \
	custom_reward_function.name=$REWARD_FN \
	+custom_reward_function.reward_kwargs.coordinates_metric=$COORDINATES_METRIC \
	+custom_reward_function.reward_kwargs.coordinates_gaussian_sigma=$COORDINATES_GAUSSIAN_SIGMA \
	+custom_reward_function.reward_kwargs.coordinates_pixel_square_size=$COORDINATES_PIXEL_SQUARE_SIZE \
	algorithm.adv_estimator=grpo \
	data.train_files=$TRAIN_FILES \
	data.val_files=$VAL_FILES \
	data.train_batch_size=$TRAIN_BATCH_SIZE \
	data.max_prompt_length=7680 \
	data.max_response_length=512 \
	data.filter_overlong_prompts=False \
	data.truncation='error' \
	data.image_key=images \
	actor_rollout_ref.model.path=$MODEL_NAME \
	actor_rollout_ref.actor.optim.lr=1e-6 \
	actor_rollout_ref.model.use_remove_padding=True \
	actor_rollout_ref.model.trust_remote_code=True \
	critic.model.trust_remote_code=True \
	reward_model.model.trust_remote_code=True \
	actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
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
	actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
	actor_rollout_ref.rollout.name=vllm \
	actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
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
	trainer.nnodes=$NUM_NODES \
	trainer.save_freq=200\
	trainer.test_freq=100 \
	trainer.s3_checkpoint_dir=$S3_CHECKPOINT_DIR \
	trainer.total_epochs=1 $@ | tee /dev/tty | grep -o "raysubmit_[a-zA-Z0-9]*" | xargs -I{} ray job logs --follow {}
