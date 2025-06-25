#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
DATA_PATH=~/data/subtask_direct_distill/mix/test/combined_with_response.parquet

# Grid search parameters
TENSOR_PARALLEL_SIZES=(1 2 4)
GPU_MEMORY_UTILS=(0.4 0.9)
BATCH_SIZES=(64 1024)

# Create results directory
mkdir -p gridsearch_results

# Initialize timing log
TIMING_LOG="gridsearch_results/timing_results.txt"
echo "timestamp,tensor_parallel_size,gpu_memory_util,batch_size,real_time,user_time,sys_time" > $TIMING_LOG

echo "Starting grid search with $(( ${#TENSOR_PARALLEL_SIZES[@]} * ${#GPU_MEMORY_UTILS[@]} * ${#BATCH_SIZES[@]} )) combinations..."

# Grid search loop
for tensor_size in "${TENSOR_PARALLEL_SIZES[@]}"; do
    for gpu_util in "${GPU_MEMORY_UTILS[@]}"; do
        for batch_size in "${BATCH_SIZES[@]}"; do
            
            # Create unique output filename
            OUTPUT_FILE="gridsearch_results/output_tp${tensor_size}_gpu${gpu_util}_bs${batch_size}.parquet"
            
            echo "----------------------------------------"
            echo "Running: tensor_parallel_size=${tensor_size}, gpu_memory_utilization=${gpu_util}, batch_size=${batch_size}"
            echo "Output: ${OUTPUT_FILE}"
            echo "----------------------------------------"
            
            # Run with timing
            TIME_OUTPUT=$(mktemp)
            /usr/bin/time -f "%e,%U,%S" -o "$TIME_OUTPUT" python3 -m orby.trainer.main_generation \
                trainer.nnodes=1 \
                trainer.n_gpus_per_node=8 \
                data.path=$DATA_PATH \
                data.prompt_key=prompt \
                data.batch_size=$batch_size \
                +data.max_prompt_length=7680 \
                data.n_samples=1 \
                data.output_path=$OUTPUT_FILE \
                model.path=$MODEL_PATH \
                rollout.temperature=0 \
                rollout.top_p=1.0 \
                rollout.prompt_length=7680 \
                rollout.response_length=512 \
                rollout.tensor_model_parallel_size=$tensor_size \
                rollout.gpu_memory_utilization=$gpu_util \
                rollout.max_num_batched_tokens=65536 \
                +rollout.limit_images=3
            
            # Parse timing results and log them
            if [ -f "$TIME_OUTPUT" ]; then
                TIMING_DATA=$(cat "$TIME_OUTPUT")
                TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
                echo "${TIMESTAMP},${tensor_size},${gpu_util},${batch_size},${TIMING_DATA}" >> $TIMING_LOG
                echo "Completed in: ${TIMING_DATA} (real,user,sys)"
                rm "$TIME_OUTPUT"
            fi
            
            echo ""
        done
    done
done

echo "Grid search completed!"
echo "Timing results saved to: $TIMING_LOG"
echo "Output files saved to: gridsearch_results/"

# Display summary
echo ""
echo "=== TIMING SUMMARY ==="
echo "tensor_parallel_size,gpu_memory_util,batch_size,real_time"
tail -n +2 "$TIMING_LOG" | cut -d',' -f2,3,4,5 | sort -t',' -k4 -n 