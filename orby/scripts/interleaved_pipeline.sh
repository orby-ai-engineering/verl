#!/bin/bash

# Function to find checkpoint with maximum steps from S3 directory
find_max_step_checkpoint() {
    local s3_dir="$1"
        
    # List all checkpoint directories and extract step numbers
    local max_steps=0
    local max_steps_checkpoint=""
    
    # Get list of checkpoint directories from S3
    local checkpoint_dirs=$(aws s3 ls "$s3_dir" | grep "global_step_" | awk '{print $2}' | sed 's|/$||')
    
    if [ -z "$checkpoint_dirs" ]; then
        echo "Error: No checkpoint directories found in $s3_dir" >&2
        return 1
    fi
    
    # Find the one with maximum steps
    for checkpoint_dir in $checkpoint_dirs; do
        # Extract step number from directory name (e.g., "global_step_100" -> 100)
        if [[ $checkpoint_dir =~ global_step_([0-9]+) ]]; then
            local step_num=${BASH_REMATCH[1]}
            
            if [ $step_num -gt $max_steps ]; then
                max_steps=$step_num
                max_steps_checkpoint=$checkpoint_dir
            fi
        fi
    done
    
    if [ -z "$max_steps_checkpoint" ]; then
        echo "Error: No valid checkpoint directories found with 'global_step_' pattern" >&2
        return 1
    fi

    echo "${s3_dir}${max_steps_checkpoint}"
}

export S3_INITIAL_SFT_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/initial_sft/

# Run initial SFT step
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=$NUM_NODES \
#     --node_rank=$NODE_RANK \
#     --master_addr=$MASTER_ADDR \
#     --master_port=${MASTER_PORT:-29500} \
#     -m orby.trainer.fsdp_sft_trainer \
#     data.train_batch_size=$SFT_TRAIN_BATCH_SIZE \
#     data.micro_batch_size_per_gpu=2 \
#     data.train_files=$SFT_TRAIN_FILES \
#     data.val_files=$SFT_VAL_FILES \
#     +data.max_prompt_length=7680 \
#     +data.max_response_length=512 \
#     +data.filter_overlong_prompts=False \
#     data.truncation='error' \
#     +data.shuffle=True \
#     data.prompt_key=prompt \
#     data.response_key=response \
#     +data.image_key=images \
#     +processor.use_fast=true \
#     +processor.trust_remote_code=true \
#     optim.lr=1e-6 \
#     model.partial_pretrain=$MODEL_NAME \
#     model.fsdp_config.cpu_offload=true \
#     model.enable_gradient_checkpointing=true \
#     +model.enable_activation_offload=true \
#     model.fsdp_config.offload_params=true \
#     +model.fsdp_config.param_offload=true \
#     trainer.default_local_dir=$S3_INITIAL_SFT_CHECKPOINT_DIR \
#     trainer.total_training_steps=null \
#     trainer.project_name=$PROJECT_NAME \
#     trainer.experiment_name=${EXPERIMENT_NAME}_initial_sft \
#     trainer.logger=[console,wandb] \
#     trainer.default_hdfs_dir=null \
#     +trainer.val_interval=100 \
#     +trainer.save_interval=100 \
#     trainer.total_epochs=1 \
#     ulysses_sequence_parallel_size=1 \
#     use_remove_padding=false \
#     +model.fsdp_config.reshard_after_forward=true \
#     +model.use_remove_padding=true \
#     model.fsdp_config.wrap_policy.min_num_params=1000000 \
#     +model.fsdp_config.optimizer_offload=true

# Find and copy the initial SFT checkpoint with maximum steps
export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$S3_INITIAL_SFT_CHECKPOINT_DIR")
echo "Found initial SFT checkpoint: $MAX_STEPS_CHECKPOINT"

if [ $? -ne 0 ]; then
    echo "Failed to find initial SFT checkpoint"
    exit 1
fi

# Copy the initial SFT checkpoint with maximum steps
export STEP_DIR=$(echo $MAX_STEPS_CHECKPOINT | grep -o "global_step_[0-9]*")
export LOCAL_SFT_CHECKPOINT=$INTERLEAVED_MODEL_DIR/initial_sft/$STEP_DIR
aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT $LOCAL_SFT_CHECKPOINT

for i in $(seq 0 $((INTERLEAVED_STEP_NUM - 1))); do
    PER_STEP_TRAIN_FILES=$INTERLEAVED_DATA_DIR/$i/train.parquet
    PER_STEP_VAL_FILES=$INTERLEAVED_DATA_DIR/$i/test.parquet
    LOCAL_OUTPUT_PARQUET=$INTERLEAVED_DATA_DIR/$i/train_rollout.parquet
    S3_OUTPUT_PARQUET=$S3_OUTPUT_DIR/$i/train_rollout.parquet

    # Start ray cluster and wait for all nodes
    bash orby/scripts/run_ray.sh $NUM_NODES

    # 1) Run rollout using the previous checkpoint
    echo "======Step $i: generating rollout data======"
    if [ "$NODE_RANK" = "0" ]; then
        echo "======Step $i: submitting rollout job on node 0======"
        ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        --no-wait \
        -- \
        python3 -u -m orby.trainer.main_generation \
            trainer.nnodes=$NUM_NODES \
            trainer.n_gpus_per_node=8 \
            data.path=$PER_STEP_TRAIN_FILES \
            data.prompt_key=prompt \
            +data.response_key=predictions \
            data.batch_size=$ROLLOUT_BATCH_SIZE \
            +data.max_prompt_length=7680 \
            +data.filter_overlong_prompts=False \
            data.output_path=$LOCAL_OUTPUT_PARQUET \
            +data.dataloader_num_workers=1 \
            model.path=$LOCAL_SFT_CHECKPOINT \
            rollout.temperature=$TEMPERATURE \
            rollout.top_p=1.0 \
            rollout.prompt_length=7680 \
            rollout.response_length=512 \
            rollout.tensor_model_parallel_size=1 \
            rollout.gpu_memory_utilization=0.9 \
            rollout.max_num_batched_tokens=65536 \
            rollout.n=$N_SAMPLES \
            +rollout.remove_multimodal_data_from_rollout=True \
            +rollout.limit_images=3 $@ | tee /dev/tty | grep -o "raysubmit_[a-zA-Z0-9]*" | xargs -I{} ray job logs --follow {}

        # Upload dataset to S3
        aws s3 cp --no-progress $LOCAL_OUTPUT_PARQUET $S3_OUTPUT_PARQUET

        # 2) Filtering step
        echo "======Step $i: submitting filtering job on node 0======"
        # Placeholder for filtering step

        # 3) Run GRPO step
        echo "======Step $i: submitting GRPO job on node 0======"
        export GRPO_EXPERIMENT_NAME=${EXPERIMENT_NAME}_${i}_grpo
        export S3_GRPO_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/${GRPO_EXPERIMENT_NAME}/
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
                data.train_files=$PER_STEP_TRAIN_FILES \
                data.val_files=$PER_STEP_VAL_FILES \
                data.train_batch_size=$GRPO_TRAIN_BATCH_SIZE \
                data.max_prompt_length=7680 \
                data.max_response_length=512 \
                data.filter_overlong_prompts=False \
                data.truncation='error' \
                data.image_key=images \
                data.shuffle=True \
                actor_rollout_ref.model.path=$LOCAL_SFT_CHECKPOINT \
                actor_rollout_ref.actor.optim.lr=1e-6 \
                actor_rollout_ref.model.use_remove_padding=True \
                actor_rollout_ref.actor.ppo_mini_batch_size=$GRPO_TRAIN_BATCH_SIZE \
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
                actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
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
                trainer.experiment_name=${GRPO_EXPERIMENT_NAME} \
                trainer.n_gpus_per_node=8 \
                trainer.nnodes=$NUM_NODES \
                trainer.save_freq=100 \
                trainer.test_freq=100 \
                trainer.s3_checkpoint_dir=$S3_GRPO_CHECKPOINT_DIR \
                trainer.total_epochs=1 $@ | tee /dev/tty | grep -o "raysubmit_[a-zA-Z0-9]*" | xargs -I{} ray job logs --follow {}

        # Stop ray cluster
        ray stop
    fi

    # Find and copy the GRPO checkpoint with maximum steps
    export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$S3_GRPO_CHECKPOINT_DIR")
    if [ $? -ne 0 ]; then
        echo "Failed to find GRPO checkpoint for step $i"
        exit 1
    fi
    
    if [ "$NODE_RANK" = "0" ]; then
        aws s3 ls $MAX_STEPS_CHECKPOINT/hf
        if [ $? -eq 0 ]; then
            echo "GRPO checkpoint is already available on S3: $MAX_STEPS_CHECKPOINT/hf"
            break
        fi
        echo "======Step $i: merging GRPO checkpoint on node 0======"
        python3 orby/scripts/model_merger.py merge \
        --backend fsdp \
        --hf_model_path Qwen/Qwen2.5-VL-7B-Instruct \
        --local_dir $MAX_STEPS_CHECKPOINT/actor \
        --target_dir $MAX_STEPS_CHECKPOINT/hf/
    fi

    export STEP_DIR=$(echo $MAX_STEPS_CHECKPOINT | grep -o "global_step_[0-9]*")
    export LOCAL_GRPO_CHECKPOINT=$INTERLEAVED_MODEL_DIR/grpo_${i}/$STEP_DIR
    while true; do
        aws s3 ls $MAX_STEPS_CHECKPOINT/hf
        if [ $? -eq 0 ]; then
            echo "GRPO checkpoint is available on S3: $MAX_STEPS_CHECKPOINT/hf"
            break
        else
            echo "Waiting for GRPO checkpoint to be available on S3..."
            sleep 10
        fi
    done

    # Wait for 60 seconds to make sure the checkpoint is fully uploaded
    sleep 60

    aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT/hf $LOCAL_GRPO_CHECKPOINT

    # 4) Run SFT step
    echo "======Step $i: running SFT======"
    export SFT_EXPERIMENT_NAME=${EXPERIMENT_NAME}_${i}_sft
    export SFT_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/${SFT_EXPERIMENT_NAME}/
    torchrun \
        --nproc_per_node=8 \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=${MASTER_PORT:-29500} \
        -m orby.trainer.fsdp_sft_trainer \
        data.train_batch_size=$SFT_TRAIN_BATCH_SIZE \
        data.micro_batch_size_per_gpu=2 \
        data.train_files=$PER_STEP_TRAIN_FILES \
        data.val_files=$PER_STEP_VAL_FILES \
        +data.max_prompt_length=7680 \
        +data.max_response_length=512 \
        +data.filter_overlong_prompts=False \
        data.truncation='error' \
        +data.shuffle=True \
        data.prompt_key=prompt \
        data.response_key=response \
        +data.image_key=images \
        +processor.use_fast=true \
        +processor.trust_remote_code=true \
        optim.lr=1e-6 \
        model.partial_pretrain=$LOCAL_GRPO_CHECKPOINT \
        model.fsdp_config.cpu_offload=true \
        model.enable_gradient_checkpointing=true \
        +model.enable_activation_offload=true \
        model.fsdp_config.offload_params=true \
        +model.fsdp_config.param_offload=true \
        trainer.default_local_dir=$SFT_CHECKPOINT_DIR \
        trainer.total_training_steps=null \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=${SFT_EXPERIMENT_NAME} \
        trainer.logger=[console,wandb] \
        trainer.default_hdfs_dir=null \
        +trainer.val_interval=100 \
        +trainer.save_interval=100 \
        trainer.total_epochs=1 \
        ulysses_sequence_parallel_size=1 \
        use_remove_padding=false \
        +model.fsdp_config.reshard_after_forward=true \
        +model.use_remove_padding=true \
        model.fsdp_config.wrap_policy.min_num_params=1000000 \
        +model.fsdp_config.optimizer_offload=true

    # Find and copy the SFT checkpoint with maximum steps
    export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$SFT_CHECKPOINT_DIR")
    if [ $? -ne 0 ]; then
        echo "Failed to find SFT checkpoint for step $i"
        exit 1
    fi
    export STEP_DIR=$(echo $MAX_STEPS_CHECKPOINT | grep -o "global_step_[0-9]*")
    export LOCAL_SFT_CHECKPOINT=$INTERLEAVED_MODEL_DIR/sft_${i}/$STEP_DIR
    aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT $LOCAL_SFT_CHECKPOINT
done