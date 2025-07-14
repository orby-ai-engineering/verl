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

extract_step_from_checkpoint_dir() {
    local checkpoint_dir="$1"
    echo $checkpoint_dir | grep -o "global_step_[0-9]*"
}

generate_rollout_data() {
    local train_files="$1"
    local output_parquet="$2"
    local checkpoint="$3"
    local temperature="$4"
    local n_samples="$5"
    local rollout_batch_size="$6"

    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env=verl/trainer/runtime_env.yaml \
        --no-wait \
        -- \
        python3 -u -m orby.trainer.main_generation \
            trainer.nnodes=$NUM_NODES \
            trainer.n_gpus_per_node=8 \
            data.path=$train_files \
            data.prompt_key=prompt \
            +data.response_key=$GENERATED_DATA_RESPONSE_KEY \
            data.batch_size=$rollout_batch_size \
            +data.max_prompt_length=7680 \
            +data.filter_overlong_prompts=False \
            data.output_path=$output_parquet \
            +data.dataloader_num_workers=1 \
            model.path=$checkpoint \
            rollout.temperature=$temperature \
            rollout.top_p=1.0 \
            rollout.prompt_length=7680 \
            rollout.response_length=512 \
            rollout.tensor_model_parallel_size=1 \
            rollout.gpu_memory_utilization=0.9 \
            rollout.max_num_batched_tokens=65536 \
            rollout.n=$n_samples \
            +rollout.remove_multimodal_data_from_rollout=True \
            +rollout.limit_images=3 | tee /dev/tty | grep -o "raysubmit_[a-zA-Z0-9]*" | xargs -I{} ray job logs --follow {}
}

sft_step() {
    local experiment_name="$1"
    local model_name="$2"
    local batch_size="$3"
    local train_files="$4"
    local val_files="$5"
    local checkpoint_dir="$6"
    local sft_lr="$7"
    local attention_dropout="$8"
    local sft_micro_batch_size_per_gpu="$9"

    torchrun \
        --nproc_per_node=8 \
        --nnodes=$NUM_NODES \
        --node_rank=$NODE_RANK \
        --master_addr=$MASTER_ADDR \
        --master_port=${MASTER_PORT:-29500} \
        -m orby.trainer.fsdp_sft_trainer \
        data.train_batch_size=$batch_size \
        data.micro_batch_size_per_gpu=$sft_micro_batch_size_per_gpu \
        data.train_files=$train_files \
        data.val_files=$val_files \
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
        optim.lr=$sft_lr \
        +model.qwen_attention_dropout=$attention_dropout \
        model.partial_pretrain=$model_name \
        model.fsdp_config.cpu_offload=true \
        model.enable_gradient_checkpointing=true \
        +model.enable_activation_offload=true \
        model.fsdp_config.offload_params=true \
        +model.fsdp_config.param_offload=true \
        trainer.default_local_dir=$checkpoint_dir \
        trainer.total_training_steps=null \
        trainer.project_name=$PROJECT_NAME \
        trainer.experiment_name=$experiment_name \
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
}

grpo_step() {
    local experiment_name="$1"
    local train_files="$2"
    local val_files="$3"
    local checkpoint="$4"
    local grpo_train_batch_size="$5"
    local s3_checkpoint_dir="$6"
    local grpo_lr="$7"
    local grpo_micro_batch_size_per_gpu="$8"

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
        data.train_files=$train_files \
        data.val_files=$val_files \
        data.train_batch_size=$grpo_train_batch_size \
        data.max_prompt_length=7680 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=False \
        data.truncation='error' \
        data.image_key=images \
        data.shuffle=True \
        actor_rollout_ref.model.path=$checkpoint \
        actor_rollout_ref.actor.optim.lr=$grpo_lr \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$grpo_train_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$grpo_micro_batch_size_per_gpu \
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
        trainer.experiment_name=$experiment_name \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=$NUM_NODES \
        trainer.save_freq=100 \
        trainer.test_freq=100 \
        trainer.s3_checkpoint_dir=$s3_checkpoint_dir \
        trainer.total_epochs=1 | tee /dev/tty | grep -o "raysubmit_[a-zA-Z0-9]*" | xargs -I{} ray job logs --follow {}
}

filter_step() {
    local input_parquet_with_rollout="$1"
    local medium_difficulty_train_files="$2"
    local hard_difficulty_train_files="$3"
    local medium_difficulty_filter_bound_str="$4"
    local hard_difficulty_filter_bound_str="$5"

    # Generate scores for each row of the input parquet with rollout
    python3 -m orby.trainer.main_eval \
        data.path=$input_parquet_with_rollout \
        data.prompt_key=prompt \
        data.response_key=$GENERATED_DATA_RESPONSE_KEY \
        +data.save_scores=True \
        custom_reward_function.path=$REWARD_FILE \
        custom_reward_function.name=$REWARD_FN \
        +custom_reward_function.reward_kwargs.coordinates_metric=$COORDINATES_METRIC \
        +custom_reward_function.reward_kwargs.coordinates_gaussian_sigma=$COORDINATES_GAUSSIAN_SIGMA \
        +custom_reward_function.reward_kwargs.coordinates_pixel_square_size=$COORDINATES_PIXEL_SQUARE_SIZE

    # Filter the input parquet with rollout based on the scores
    python3 -m orby.trainer.main_reward_filter \
        data.path=$input_parquet_with_rollout \
        data.medium_difficulty_output_path=$medium_difficulty_train_files \
        data.hard_difficulty_output_path=$hard_difficulty_train_files \
        data.medium_difficulty_filter_bound=$medium_difficulty_filter_bound_str \
        data.hard_difficulty_filter_bound=$hard_difficulty_filter_bound_str \
        data.balance_should_end=true \
        data.should_end_column="reward_model.ground_truth.should_end" \
        data.reward_score_column=$REWARD_SCORE_COLUMN
}

function merge_checkpoint() {
    local max_steps_checkpoint="$1"
    aws s3 ls $max_steps_checkpoint/hf
    if [ $? -eq 0 ]; then
        echo "GRPO checkpoint is already available on S3: $max_steps_checkpoint/hf"
        return 0
    fi
    python3 orby/scripts/model_merger.py merge \
    --backend fsdp \
    --hf_model_path Qwen/Qwen2.5-VL-7B-Instruct \
    --local_dir $max_steps_checkpoint/actor \
    --target_dir $max_steps_checkpoint/hf/
}

function wait_for_hf_checkpoint() {
    local max_steps_checkpoint="$1"
    while true; do
        aws s3 ls $max_steps_checkpoint
        if [ $? -eq 0 ]; then
            echo "Checkpoint is available on S3: $max_steps_checkpoint"
            break
        else
            echo "Waiting for checkpoint to be available on S3..."
            sleep 10
        fi
    done
}


echo "======Initial SFT step======"
export S3_INITIAL_SFT_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/initial_sft/
export INITIAL_SFT_EXPERIMENT_NAME=${EXPERIMENT_NAME}_initial_sft

# Run initial SFT step
sft_step $INITIAL_SFT_EXPERIMENT_NAME \
    $MODEL_NAME \
    $SFT_TRAIN_BATCH_SIZE \
    $INITIAL_SFT_TRAIN_FILES \
    $SHARED_VAL_FILES \
    $S3_INITIAL_SFT_CHECKPOINT_DIR \
    $SFT_LR \
    $ATTENTION_DROPOUT \
    $SFT_MICRO_BATCH_SIZE_PER_GPU

# Find and copy the initial SFT checkpoint with maximum steps
export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$S3_INITIAL_SFT_CHECKPOINT_DIR")
echo "Found initial SFT checkpoint: $MAX_STEPS_CHECKPOINT"

# Copy the initial SFT checkpoint with maximum steps
export STEP_DIR=$(extract_step_from_checkpoint_dir $MAX_STEPS_CHECKPOINT)
export LOCAL_SFT_CHECKPOINT=$LOCAL_MODEL_DIR/initial_sft/$STEP_DIR
aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT $LOCAL_SFT_CHECKPOINT

# Main loop
for i in $(seq 0 $((INTERLEAVED_STEP_NUM - 1))); do
    PER_STEP_TRAIN_FILES=$LOCAL_DATA_DIR/$i/train.parquet
    # We use shared validation files for all steps.
    # PER_STEP_VAL_FILES=$LOCAL_DATA_DIR/$i/test.parquet
    LOCAL_OUTPUT_PARQUET=$LOCAL_DATA_DIR/$i/train_rollout.parquet
    ROLLOUT_OUTPUT_PARQUET=$ROLLOUT_OUTPUT_DIR/$i/train_rollout.parquet

    # Start ray cluster and wait for all nodes
    bash orby/scripts/run_ray.sh $NUM_NODES

    # Export vars for all nodes
    export GRPO_EXPERIMENT_NAME=${EXPERIMENT_NAME}_${i}_grpo
    export S3_GRPO_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/${GRPO_EXPERIMENT_NAME}/

    # 1) Run rollout using the previous checkpoint
    echo "======Step $i: generating rollout data======"

    if [ "$NODE_RANK" = "0" ]; then
        echo "======Step $i: submitting rollout job on node 0======"
        generate_rollout_data $PER_STEP_TRAIN_FILES \
        $LOCAL_OUTPUT_PARQUET \
        $LOCAL_SFT_CHECKPOINT \
        $TEMPERATURE \
        $N_SAMPLES \
        $ROLLOUT_BATCH_SIZE

        # Upload dataset to S3
        aws s3 cp --no-progress $LOCAL_OUTPUT_PARQUET $ROLLOUT_OUTPUT_PARQUET

        # 2) Filtering step
        echo "======Step $i: submitting filtering job on node 0======"
        filter_step $LOCAL_OUTPUT_PARQUET \
        $PER_STEP_GRPO_TRAIN_FILES \
        $PER_STEP_SFT_TRAIN_FILES \
        $MEDIUM_DIFFICULTY_FILTER_BOUND_STR \
        $HARD_DIFFICULTY_FILTER_BOUND_STR

        # 3) Run GRPO step
        echo "======Step $i: submitting GRPO job on node 0======"
        grpo_step $GRPO_EXPERIMENT_NAME \
        $PER_STEP_GRPO_TRAIN_FILES \
        $SHARED_VAL_FILES \
        $LOCAL_SFT_CHECKPOINT \
        $GRPO_TRAIN_BATCH_SIZE \
        $S3_GRPO_CHECKPOINT_DIR \
        $GRPO_LR \
        $GRPO_MICRO_BATCH_SIZE_PER_GPU

        # Stop ray cluster
        ray stop
    fi

    # Find the GRPO checkpoint with maximum steps
    export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$S3_GRPO_CHECKPOINT_DIR")

    if [ "$NODE_RANK" = "0" ]; then
        echo "======Step $i: merging GRPO checkpoint on node 0======"
        merge_checkpoint $MAX_STEPS_CHECKPOINT
    fi

    # Wait for the checkpoint on other nodes.
    wait_for_hf_checkpoint $MAX_STEPS_CHECKPOINT/hf
    # Wait for 300 seconds to make sure the checkpoint is fully uploaded
    sleep 300

    # Download the merged checkpoint
    export STEP_DIR=$(extract_step_from_checkpoint_dir $MAX_STEPS_CHECKPOINT)
    export LOCAL_GRPO_CHECKPOINT=$LOCAL_MODEL_DIR/grpo_${i}/$STEP_DIR
    aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT/hf $LOCAL_GRPO_CHECKPOINT

    # 4) Run SFT step
    echo "======Step $i: running SFT======"
    export SFT_EXPERIMENT_NAME=${EXPERIMENT_NAME}_${i}_sft
    export SFT_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/${SFT_EXPERIMENT_NAME}/
    sft_step $SFT_EXPERIMENT_NAME \
    $LOCAL_GRPO_CHECKPOINT \
    $SFT_TRAIN_BATCH_SIZE \
    $PER_STEP_SFT_TRAIN_FILES \
    $SHARED_VAL_FILES \
    $SFT_CHECKPOINT_DIR \
    $SFT_LR \
    $ATTENTION_DROPOUT \
    $SFT_MICRO_BATCH_SIZE_PER_GPU

    # Find and copy the SFT checkpoint with maximum steps
    export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$SFT_CHECKPOINT_DIR")
    export STEP_DIR=$(extract_step_from_checkpoint_dir $MAX_STEPS_CHECKPOINT)
    export LOCAL_SFT_CHECKPOINT=$LOCAL_MODEL_DIR/sft_${i}/$STEP_DIR
    aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT $LOCAL_SFT_CHECKPOINT
done