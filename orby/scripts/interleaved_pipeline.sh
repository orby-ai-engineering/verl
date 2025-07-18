#!/bin/bash
set -u # Exit on undefined variable

# Clean up synchronization flags on in case of resume
if [ "$NODE_RANK" = "0" ]; then
    aws s3 rm --recursive "$S3_CHECKPOINT_DIR/sync_flags/" >/dev/null 2>&1 || true
fi

# Create all directories
mkdir -p $LOCAL_DATA_DIR
mkdir -p $LOCAL_MODEL_DIR
mkdir -p $LOCAL_EVAL_DIR

# Function to find checkpoint with maximum steps from S3 directory
find_max_step_checkpoint() {
    local s3_dir="$1"

    # List all checkpoint directories and extract step numbers
    local max_steps=0
    local max_steps_checkpoint=""
    
    # Get list of checkpoint directories from S3
    local checkpoint_dirs=$(aws s3 ls "$s3_dir" | grep "global_step_" | awk '{print $2}' | sed 's|/$||')
    
    if [ -z "$checkpoint_dirs" ]; then
        echo "TOP LEVEL - Error: No checkpoint directories found in $s3_dir" >&2
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
        echo "TOP LEVEL - Error: No valid checkpoint directories found with 'global_step_' pattern" >&2
        return 1
    fi

    echo "${s3_dir}${max_steps_checkpoint}" # Return the checkpoint path
}

# Node synchronization functions
signal_step_complete() {
    local step_name="$1"
    local sync_flag="$S3_CHECKPOINT_DIR/sync_flags/${step_name}_complete.flag"
    
    if [ "$NODE_RANK" = "0" ]; then
        echo "Node 0 signaling completion of: $step_name"
        echo "$(date)" | aws s3 cp - "$sync_flag"
    fi
}

wait_for_step_complete() {
    local step_name="$1"
    local sync_flag="$S3_CHECKPOINT_DIR/sync_flags/${step_name}_complete.flag"
    
    if [ "$NODE_RANK" != "0" ]; then
        echo "Node $NODE_RANK waiting for step: $step_name"
        while ! aws s3 ls "$sync_flag" >/dev/null 2>&1; do
            echo "Node $NODE_RANK still waiting for $step_name..."
            sleep 5
        done
        echo "Node $NODE_RANK: $step_name completed, proceeding..."
    fi
}

# Combined function for cleaner code
run_on_node0_and_sync() {
    local step_name="$1"
    shift  # Remove first argument, rest are the commands to run
    
    if [ "$NODE_RANK" = "0" ]; then
        echo "Node 0 executing: $step_name"
        "$@"  # Execute the remaining arguments as a command
        signal_step_complete "$step_name"
    fi
    
    wait_for_step_complete "$step_name"
}

# Helper functions
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
        python3 -m orby.trainer.main_generation \
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
    local warmup_steps_ratio="${10}"
    local lr_scheduler="${11}"

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
        optim.warmup_steps_ratio=$warmup_steps_ratio \
        optim.lr_scheduler=$lr_scheduler \
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
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
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
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
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

eval_step() {
    local eval_data_path="$1"
    local model_path="$2"
    local current_index="$3"

    # Generation
    python3 -m orby.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=8 \
        data.path=$eval_data_path \
        data.prompt_key=prompt \
        data.batch_size=1024 \
        +data.max_prompt_length=7680 \
        +data.filter_overlong_prompts=false \
        data.n_samples=1 \
        data.output_path=$EVAL_OUTPUT_FILE \
        model.path=$model_path \
        rollout.temperature=0 \
        rollout.top_p=1.0 \
        rollout.prompt_length=7680 \
        rollout.response_length=512 \
        rollout.tensor_model_parallel_size=1 \
        rollout.gpu_memory_utilization=0.9 \
        rollout.max_num_batched_tokens=65536 \
        +rollout.limit_images=3

    # Evaluation
    python3 -m orby.trainer.main_eval \
        data.path=$EVAL_OUTPUT_FILE \
        data.prompt_key=prompt \
        data.response_key=responses \
        +data.save_scores=false \
        +data.interleave.store_to_local_parquet=true \
        +data.interleave.local_parquet_path=$LOCAL_EVAL_RESULT_FILE \
        +data.interleave.current_index=$current_index \
        custom_reward_function.path=$REWARD_FILE \
        custom_reward_function.name=$EVAL_REWARD_FN \
        +custom_reward_function.reward_kwargs.coordinates_metric=$COORDINATES_METRIC \
        +custom_reward_function.reward_kwargs.coordinates_gaussian_sigma=$COORDINATES_GAUSSIAN_SIGMA \
        +custom_reward_function.reward_kwargs.coordinates_pixel_square_size=$COORDINATES_PIXEL_SQUARE_SIZE
}

filter_step() {
    local input_parquet_with_rollout="$1"
    local medium_difficulty_train_files="$2"
    local hard_difficulty_train_files="$3"
    local medium_difficulty_filter_upper_bound="$4"
    local medium_difficulty_filter_lower_bound="$5"
    local hard_difficulty_filter_upper_bound="$6"
    local hard_difficulty_filter_lower_bound="$7"

    echo "TOP LEVEL - Step 1.$i.1.0: generating scores ======================================================="

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

    echo "TOP LEVEL - Step 1.$i.1.1: filtering by score ================================================="

    input_num_rows=$(parquet-tools inspect $input_parquet_with_rollout | grep 'num_rows' | awk '{print $2}')
    echo "TOP LEVEL - Before filtering, the full rollout parquet has $input_num_rows rows"

    # Filter the input parquet with rollout based on the scores
    python3 -m orby.trainer.main_reward_filter \
        data.path=$input_parquet_with_rollout \
        data.medium_difficulty_output_path=$medium_difficulty_train_files \
        data.hard_difficulty_output_path=$hard_difficulty_train_files \
        data.medium_difficulty_filter_upper_bound=$medium_difficulty_filter_upper_bound \
        data.medium_difficulty_filter_lower_bound=$medium_difficulty_filter_lower_bound \
        data.hard_difficulty_filter_upper_bound=$hard_difficulty_filter_upper_bound \
        data.hard_difficulty_filter_lower_bound=$hard_difficulty_filter_lower_bound \
        data.balance_should_end=true \
        data.should_end_column="reward_model.ground_truth.should_end" \
        data.reward_score_column=$REWARD_SCORE_COLUMN

    medium_difficulty_num_rows=$(parquet-tools inspect $medium_difficulty_train_files | grep 'num_rows' | awk '{print $2}')
    hard_difficulty_num_rows=$(parquet-tools inspect $hard_difficulty_train_files | grep 'num_rows' | awk '{print $2}')
    echo "TOP LEVEL - After filtering, the medium difficulty parquet has $medium_difficulty_num_rows rows"
    echo "TOP LEVEL - After filtering, the hard difficulty parquet has $hard_difficulty_num_rows rows"
}

function merge_checkpoint() {
    local max_steps_checkpoint="$1"

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
            echo "TOP LEVEL - Checkpoint is available on S3: $max_steps_checkpoint"
            break
        else
            echo "TOP LEVEL - Waiting for checkpoint to be available on S3..."
            sleep 10
        fi
    done
}


# Start of the pipeline
echo "TOP LEVEL - Step 0: Initial SFT step ==============================================================="
export INITIAL_SFT_EXPERIMENT_NAME=${EXPERIMENT_NAME}_initial_sft
export S3_INITIAL_SFT_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/initial_sft/

# If the S3_INITIAL_SFT_CHECKPOINT_DIR is not empty, we skip the initial SFT step (resume)
if aws s3 ls "$S3_INITIAL_SFT_CHECKPOINT_DIR" >/dev/null 2>&1; then
    echo "TOP LEVEL - Skip initial SFT step due to existing checkpoint (resume)"
    export S3_INIT_SFT_CHECKPOINT=$(find_max_step_checkpoint "$S3_INITIAL_SFT_CHECKPOINT_DIR")
elif [ -z "$BASE_SFT_CHECKPOINT" ]; then
    # If BASE_SFT_CHECKPOINT is not set, we train from scratch
    echo "TOP LEVEL - Step 0.0b: training from scratch ======================================================="
    # Download model
    python3 -c "import transformers; transformers.pipeline(model='$MODEL_NAME', device='cpu')"

    # Run initial SFT step
    sft_step $INITIAL_SFT_EXPERIMENT_NAME \
        $MODEL_NAME \
        $SFT_TRAIN_BATCH_SIZE \
        $INITIAL_SFT_TRAIN_FILES \
        $SHARED_VAL_FILES \
        $S3_INITIAL_SFT_CHECKPOINT_DIR \
        $SFT_LR \
        $ATTENTION_DROPOUT \
        $SFT_MICRO_BATCH_SIZE_PER_GPU \
        $INITIAL_SFT_WARMUP_STEPS_RATIO \
        $INITIAL_LR_SCHEDULER

    export S3_INIT_SFT_CHECKPOINT=$(find_max_step_checkpoint "$S3_INITIAL_SFT_CHECKPOINT_DIR")
else
    # Otherwise we download the provided initial checkpoint
    echo "TOP LEVEL - Step 0.0c: downloading initial SFT checkpoint =========================================="
    export STEP_DIR=$(extract_step_from_checkpoint_dir $BASE_SFT_CHECKPOINT)
    export S3_INIT_SFT_CHECKPOINT=${S3_INITIAL_SFT_CHECKPOINT_DIR}${STEP_DIR}
    # Copy the initial SFT checkpoint with maximum steps; only make one call on node 0
    run_on_node0_and_sync "base_sft_checkpoint_copy" \
        aws s3 cp --no-progress --recursive $BASE_SFT_CHECKPOINT $S3_INIT_SFT_CHECKPOINT
    sleep 60 # Wait to make sure the initial SFT checkpoint is fully uploaded
fi

echo "TOP LEVEL - Collected initial SFT checkpoint: $S3_INIT_SFT_CHECKPOINT"

# Copy the initial SFT checkpoint with maximum steps
export STEP_DIR=$(extract_step_from_checkpoint_dir $S3_INIT_SFT_CHECKPOINT)
export LOCAL_SFT_CHECKPOINT=$LOCAL_MODEL_DIR/initial_sft/$STEP_DIR
# Download the SFT checkpoint on all nodes
aws s3 cp --no-progress --recursive $S3_INIT_SFT_CHECKPOINT $LOCAL_SFT_CHECKPOINT

# Evaluation
echo "TOP LEVEL - Step 0.1: evaluating initial SFT checkpoint ============================================"
run_on_node0_and_sync "initial_evaluation" \
    eval_step $SHARED_VAL_FILES $LOCAL_SFT_CHECKPOINT "initial_sft"

# Main loop
echo "TOP LEVEL - Step 1: Main loop ======================================================================"
for i in $(seq 0 $((INTERLEAVED_STEP_NUM - 1))); do
    PER_STEP_TRAIN_FILES=$LOCAL_DATA_DIR/$i/train.parquet
    # We use shared validation files for all steps.
    # PER_STEP_VAL_FILES=$LOCAL_DATA_DIR/$i/test.parquet
    LOCAL_OUTPUT_PARQUET=$LOCAL_DATA_DIR/$i/train_rollout.parquet
    ROLLOUT_OUTPUT_PARQUET=$S3_ROLLOUT_OUTPUT_DIR/$i/train_rollout.parquet

    # We store the filtered datasets for each step and upload them to S3 for workers to download
    LOCAL_PER_STEP_GRPO_TRAIN_FILES=$LOCAL_DATA_DIR/$i/grpo_train.parquet
    LOCAL_PER_STEP_SFT_TRAIN_FILES=$LOCAL_DATA_DIR/$i/sft_train.parquet
    S3_PER_STEP_GRPO_TRAIN_FILES=$S3_ROLLOUT_OUTPUT_DIR/$i/grpo_train.parquet
    S3_PER_STEP_SFT_TRAIN_FILES=$S3_ROLLOUT_OUTPUT_DIR/$i/sft_train.parquet

    # Start ray cluster and wait for all nodes
    echo "TOP LEVEL - Step 1.$i.0: starting ray cluster ======================================================"
    bash orby/scripts/run_ray.sh $NUM_NODES

    # 1) Run rollout using the previous checkpoint
    echo "TOP LEVEL - Step 1.$i.1: generating rollout data ==================================================="
    if aws s3 ls "$ROLLOUT_OUTPUT_PARQUET" >/dev/null 2>&1; then
        # If the rollout output parquet already exists on S3, we skip the rollout step (resume)
        echo "TOP LEVEL - Skip rollout step due to existing rollout data (resume)"
        if [ "$NODE_RANK" = "0" ]; then
            aws s3 cp --no-progress $ROLLOUT_OUTPUT_PARQUET $LOCAL_OUTPUT_PARQUET
        fi
    else
        # Otherwise we generate the rollout data
        if [ "$NODE_RANK" = "0" ]; then
            generate_rollout_data $PER_STEP_TRAIN_FILES \
                $LOCAL_OUTPUT_PARQUET \
                $LOCAL_SFT_CHECKPOINT \
                $TEMPERATURE \
                $N_SAMPLES \
                $ROLLOUT_BATCH_SIZE

            # Upload dataset to S3; only make one call on node 0
            aws s3 cp --no-progress $LOCAL_OUTPUT_PARQUET $ROLLOUT_OUTPUT_PARQUET
        fi
        # Note: No synchronization needed here - Ray manages the distributed execution
    fi

    # 2) Filtering step
    echo "TOP LEVEL - Step 1.$i.2: filter by difficulty on node 0 ============================================"
    if aws s3 ls "$S3_PER_STEP_GRPO_TRAIN_FILES" >/dev/null 2>&1; then
        # If the filtered datasets already exist on S3, we skip the filtering step (resume)
        echo "TOP LEVEL - Skip filtering step due to existing filtered datasets (resume)"
        if [ "$NODE_RANK" = "0" ]; then
            aws s3 cp --no-progress $S3_PER_STEP_GRPO_TRAIN_FILES $LOCAL_PER_STEP_GRPO_TRAIN_FILES
            aws s3 cp --no-progress $S3_PER_STEP_SFT_TRAIN_FILES $LOCAL_PER_STEP_SFT_TRAIN_FILES
        fi
    else
        # Otherwise we filter the datasets
        if [ "$NODE_RANK" = "0" ]; then
            filter_step $LOCAL_OUTPUT_PARQUET \
                $LOCAL_PER_STEP_GRPO_TRAIN_FILES \
                $LOCAL_PER_STEP_SFT_TRAIN_FILES \
                $MEDIUM_DIFFICULTY_FILTER_UPPER_BOUND \
                $MEDIUM_DIFFICULTY_FILTER_LOWER_BOUND \
                $HARD_DIFFICULTY_FILTER_UPPER_BOUND \
                $HARD_DIFFICULTY_FILTER_LOWER_BOUND

            # Upload dataset to S3; only make one call on node 0
            aws s3 cp --no-progress $LOCAL_PER_STEP_GRPO_TRAIN_FILES $S3_PER_STEP_GRPO_TRAIN_FILES
            aws s3 cp --no-progress $LOCAL_PER_STEP_SFT_TRAIN_FILES $S3_PER_STEP_SFT_TRAIN_FILES
        fi
        # Note: No synchronization needed here - Ray manages the distributed execution
    fi

    # 3) Run GRPO step
    echo "TOP LEVEL - Step 1.$i.3: submitting GRPO job on node 0 ============================================="
    # Export vars for all nodes
    export GRPO_EXPERIMENT_NAME=${EXPERIMENT_NAME}_${i}_grpo
    export S3_GRPO_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/${i}/grpo/

    if aws s3 ls "$S3_GRPO_CHECKPOINT_DIR" >/dev/null 2>&1; then
        # If the GRPO checkpoint already exists on S3, we skip the GRPO step (resume)
        echo "TOP LEVEL - Skip GRPO step due to existing GRPO checkpoint (resume)"
    else
        # Otherwise we run the GRPO step
        if [ "$NODE_RANK" = "0" ]; then
            grpo_step $GRPO_EXPERIMENT_NAME \
                $LOCAL_PER_STEP_GRPO_TRAIN_FILES \
                $SHARED_VAL_FILES \
                $LOCAL_SFT_CHECKPOINT \
                $GRPO_TRAIN_BATCH_SIZE \
                $S3_GRPO_CHECKPOINT_DIR \
                $GRPO_LR \
                $GRPO_MICRO_BATCH_SIZE_PER_GPU
        fi
        # Note: No synchronization needed here - Ray manages the distributed execution
    fi

    # Stop ray cluster
    if [ "$NODE_RANK" = "0" ]; then
        ray stop
    fi

    echo "TOP LEVEL - Step 1.$i.4: merging GRPO checkpoint on node 0 ========================================="
    # Find the GRPO checkpoint with maximum steps
    export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$S3_GRPO_CHECKPOINT_DIR")
    export MAX_STEPS_CHECKPOINT_HF="${MAX_STEPS_CHECKPOINT}/hf"
    export STEP_DIR=$(extract_step_from_checkpoint_dir $MAX_STEPS_CHECKPOINT)

    # If the GRPO checkpoint already exists on S3, we skip the merging step (resume)
    if aws s3 ls "$MAX_STEPS_CHECKPOINT_HF" >/dev/null 2>&1; then
        echo "TOP LEVEL - Skip merging step due to existing GRPO checkpoint (resume)"
    else
        # Otherwise we merge the GRPO checkpoint
        run_on_node0_and_sync "checkpoint_merge_$i" \
            merge_checkpoint $MAX_STEPS_CHECKPOINT

        # Wait for the checkpoint on other nodes.
        wait_for_hf_checkpoint $MAX_STEPS_CHECKPOINT_HF
        # Wait for 300 seconds to make sure the checkpoint is fully uploaded
        sleep 300
    fi

    # Download the GRPO checkpoint on all nodes
    export LOCAL_GRPO_CHECKPOINT=$LOCAL_MODEL_DIR/grpo_${i}/$STEP_DIR
    aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT_HF $LOCAL_GRPO_CHECKPOINT

    # Evaluation
    echo "TOP LEVEL - Step 1.$i.5: evaluating GRPO checkpoint $i on node 0 ==================================="
    run_on_node0_and_sync "grpo_eval_$i" \
        eval_step $SHARED_VAL_FILES $LOCAL_GRPO_CHECKPOINT "${i}_grpo"

    # 4) Run SFT step
    echo "TOP LEVEL - Step 1.$i.6: running SFT ==============================================================="
    export SFT_EXPERIMENT_NAME=${EXPERIMENT_NAME}_${i}_sft
    export SFT_CHECKPOINT_DIR=$S3_CHECKPOINT_DIR/${i}/sft/

    if aws s3 ls "$SFT_CHECKPOINT_DIR" >/dev/null 2>&1; then
        echo "TOP LEVEL - Skip SFT step due to existing SFT checkpoint (resume)"
    else
        # Otherwise we run the SFT step
        # Download the SFT training dataset on all nodes
        aws s3 cp --no-progress $S3_PER_STEP_SFT_TRAIN_FILES $LOCAL_PER_STEP_SFT_TRAIN_FILES

        sft_step $SFT_EXPERIMENT_NAME \
            $LOCAL_GRPO_CHECKPOINT \
            $SFT_TRAIN_BATCH_SIZE \
            $LOCAL_PER_STEP_SFT_TRAIN_FILES \
            $SHARED_VAL_FILES \
            $SFT_CHECKPOINT_DIR \
            $STEP_SFT_LR \
            $ATTENTION_DROPOUT \
            $SFT_MICRO_BATCH_SIZE_PER_GPU \
            $STEP_SFT_WARMUP_STEPS_RATIO \
            $STEP_SFT_LR_SCHEDULER
    fi

    # Find and copy the SFT checkpoint with maximum steps
    export MAX_STEPS_CHECKPOINT=$(find_max_step_checkpoint "$SFT_CHECKPOINT_DIR")
    export STEP_DIR=$(extract_step_from_checkpoint_dir $MAX_STEPS_CHECKPOINT)
    export LOCAL_SFT_CHECKPOINT=$LOCAL_MODEL_DIR/sft_${i}/$STEP_DIR
    # Download the SFT checkpoint on all nodes
    aws s3 cp --no-progress --recursive $MAX_STEPS_CHECKPOINT $LOCAL_SFT_CHECKPOINT

    # evaluation
    echo "TOP LEVEL - Step 1.$i.7: evaluating SFT checkpoint $i on node 0 ===================================="
    run_on_node0_and_sync "sft_eval_$i" \
        eval_step $SHARED_VAL_FILES $LOCAL_SFT_CHECKPOINT "${i}_sft"
done

# Final report
echo "TOP LEVEL - Step 2: report results ================================================================="
# Upload the evaluation results to S3 on node 0
if [ "$NODE_RANK" = "0" ]; then
    aws s3 cp --no-progress $LOCAL_EVAL_RESULT_FILE $S3_EVAL_RESULT_FILE
    # Clean up synchronization flags
    sleep 60 # Wait to make sure worker nodes sees the final eval complete flag
    echo "TOP LEVEL - Cleaning up synchronization flags..."
    aws s3 rm --recursive "$S3_CHECKPOINT_DIR/sync_flags/" >/dev/null 2>&1 || true
fi    

echo "All training rounds are done."
echo "All checkpoints are available at: $S3_CHECKPOINT_DIR"
echo "All rollout data are available at: $S3_ROLLOUT_OUTPUT_DIR"
echo "All evaluation results are available at: $S3_EVAL_RESULT_DIR"

echo "TOP LEVEL - ALL DONE ==============================================================================="

exit 0
