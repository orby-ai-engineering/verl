set -e

## How to run:
# bash orby/scripts/eval_screenspot.sh --version screenspot
# bash orby/scripts/eval_screenspot.sh --version screenspot_v2
# bash orby/scripts/eval_screenspot.sh --version screenspot_pro
# bash orby/scripts/eval_screenspot.sh --version screenspot_sft
# bash orby/scripts/eval_screenspot.sh --version screenspot_v2_sft
# bash orby/scripts/eval_screenspot.sh --version screenspot_pro_sft
# bash orby/scripts/eval_screenspot.sh --version screenspot_subtask
# bash orby/scripts/eval_screenspot.sh --version screenspot_v2_subtask
# bash orby/scripts/eval_screenspot.sh --version screenspot_pro_subtask

## If you want to run 72B model, you need to run the following command:
# bash orby/scripts/eval_screenspot.sh --version screenspot --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_v2 --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_pro --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_sft --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_v2_sft --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_pro_sft --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_subtask --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_v2_subtask --model_size 72
# bash orby/scripts/eval_screenspot.sh --version screenspot_pro_subtask --model_size 72

# Default values
DATASET_VERSION="screenspot"
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
MODEL_SIZE=7
REWARD_FILE=orby/reward/screenspot.py
REWARD_FN=reward_func
OUTPUT_FILE=result-test-output-1.parquet
PROMPT_FORMAT="qwen"
BATCH_SIZE=256
TENSOR_MODEL_PARALLEL_SIZE=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --version)
            DATASET_VERSION="$2"
            shift 2
            ;;
        --model_name)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to handle errors and exit with proper code
handle_error() {
    local exit_code=$?
    echo "ERROR: Script failed with exit code $exit_code"
    echo "Failed at line: $1"
    exit $exit_code
}

# Set error handling
trap 'handle_error $LINENO' ERR

echo "Starting evaluation with:"
echo "  Dataset version: $DATASET_VERSION"
echo "  Model path: $MODEL_PATH"
echo "  Model size: $MODEL_SIZE"

# Set dataset-specific variables
case $DATASET_VERSION in
    "screenspot")
        DATA_PATH=~/data/screenspot
        PARQUET_PATTERN="test.parquet"
        ;;
    "screenspot_v2")
        DATA_PATH=~/data/screenspot_v2
        PARQUET_PATTERN="test.parquet"
        ;;
    "screenspot_pro")
        DATA_PATH=~/data/screenspot_pro
        PARQUET_PATTERN="test.parquet"
        ;;
    "screenspot_subtask")
        DATA_PATH=~/data/screenspot_subtask
        PARQUET_PATTERN="test.parquet"
        PROMPT_FORMAT="subtask"
        ;;
    "screenspot_v2_subtask")
        DATA_PATH=~/data/screenspot_v2_subtask
        PARQUET_PATTERN="test.parquet"
        PROMPT_FORMAT="subtask"
        ;;
    "screenspot_pro_subtask")
        DATA_PATH=~/data/screenspot_pro_subtask
        PARQUET_PATTERN="test.parquet"
        PROMPT_FORMAT="subtask"
        BATCH_SIZE=32
        ;;
    "screenspot_sft")
        DATA_PATH=~/data/screenspot_sft
        PARQUET_PATTERN="test.parquet"
        PROMPT_FORMAT="sft"
        ;;
    "screenspot_v2_sft")
        DATA_PATH=~/data/screenspot_v2_sft
        PARQUET_PATTERN="test.parquet"
        PROMPT_FORMAT="sft"
        ;;
    "screenspot_pro_sft")
        DATA_PATH=~/data/screenspot_pro_sft
        PARQUET_PATTERN="test.parquet"
        PROMPT_FORMAT="sft"
        ;;
    *)
        echo "Invalid dataset version: $DATASET_VERSION"
        echo "Available versions: screenspot, screenspot_v2, screenspot_pro, screenspot_sft, screenspot_v2_sft, screenspot_pro_sft"
        exit 1
        ;;
esac

# TODO: BUG FIX: 72B model starts the generation but fails after couple of hours with a ray/vllm error.
if [ $MODEL_SIZE -eq 72 ]; then
    TENSOR_MODEL_PARALLEL_SIZE=8
    BATCH_SIZE=1
fi

echo "Using dataset version: $DATASET_VERSION"
echo "Data path: $DATA_PATH"
echo "Using Batch Size: $BATCH_SIZE"
echo "Using Tensor Model Parallel Size: $TENSOR_MODEL_PARALLEL_SIZE"

# Check if parquet files already exist
if ls $DATA_PATH/$PARQUET_PATTERN 1> /dev/null 2>&1; then
    echo "Parquet files already exist, skipping conversion..."
else
    echo "Converting dataset..."
    case $DATASET_VERSION in
        "screenspot")
            python3 -m orby.data.convert_screenspot --prompt_format $PROMPT_FORMAT
            ;;
        "screenspot_v2")
            huggingface-cli download OS-Copilot/ScreenSpot-v2 --repo-type dataset --local-dir=$DATA_PATH
            cd $DATA_PATH
            unzip screenspotv2_image.zip
            cd -
            python orby/data/convert_screenspot_v2.py --image_dir=$DATA_PATH/screenspotv2_image/ --prompt_format $PROMPT_FORMAT
            ;;
        "screenspot_pro")
            huggingface-cli download likaixin/ScreenSpot-Pro --repo-type dataset --local-dir=$DATA_PATH
            python orby/data/convert_screenspot_pro.py --prompt_format $PROMPT_FORMAT
            ;;
        "screenspot_sft")
            python3 -m orby.data.convert_screenspot --local_dir $DATA_PATH --prompt_format $PROMPT_FORMAT
            ;;
        "screenspot_v2_sft")
            huggingface-cli download OS-Copilot/ScreenSpot-v2 --repo-type dataset --local-dir=$DATA_PATH
            cd $DATA_PATH
            unzip screenspotv2_image.zip
            cd -
            python orby/data/convert_screenspot_v2.py --local_dir $DATA_PATH --image_dir=$DATA_PATH/screenspotv2_image/ --prompt_format "$PROMPT_FORMAT"
            ;;
        "screenspot_pro_sft")
            huggingface-cli download likaixin/ScreenSpot-Pro --repo-type dataset --local-dir="$DATA_PATH"
            python orby/data/convert_screenspot_pro.py --local_dir "$DATA_PATH" --image_dir="$DATA_PATH/images/" --annotations_dir="$DATA_PATH/annotations/" --prompt_format "$PROMPT_FORMAT"
            ;;
        "screenspot_subtask")
            python3 -m orby.data.convert_screenspot --local_dir $DATA_PATH --prompt_format $PROMPT_FORMAT
            ;;
        "screenspot_v2_subtask")
            huggingface-cli download OS-Copilot/ScreenSpot-v2 --repo-type dataset --local-dir=$DATA_PATH
            cd $DATA_PATH
            unzip screenspotv2_image.zip
            cd -
            python orby/data/convert_screenspot_v2.py --local_dir $DATA_PATH --image_dir=$DATA_PATH/screenspotv2_image/ --prompt_format "$PROMPT_FORMAT"
            ;;
        "screenspot_pro_subtask")
            huggingface-cli download likaixin/ScreenSpot-Pro --repo-type dataset --local-dir="$DATA_PATH"
            python orby/data/convert_screenspot_pro.py --local_dir "$DATA_PATH" --image_dir="$DATA_PATH/images/" --annotations_dir="$DATA_PATH/annotations/" --prompt_format "$PROMPT_FORMAT"
            ;;
    esac
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Dataset conversion failed"
        exit 1
    fi
fi

echo "Starting generation phase..."

# Generation
# Screenspot pro has example with more than 16k tokens.
python3 -m orby.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/$PARQUET_PATTERN \
    data.prompt_key=prompt \
    data.batch_size=$BATCH_SIZE \
    +data.max_prompt_length=20000 \
    +data.image_key=images \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/$OUTPUT_FILE \
    model.path=$MODEL_PATH \
    rollout.temperature=0 \
    rollout.top_p=1.0 \
    rollout.prompt_length=20000 \
    rollout.response_length=256 \
    rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    rollout.gpu_memory_utilization=0.7 \
    rollout.max_num_batched_tokens=65536

if [ $? -ne 0 ]; then
    echo "ERROR: Generation phase failed"
    exit 1
fi

echo "Generation completed successfully. Starting evaluation phase..."

# Evaluation
export UPLOAD_TO_WANDB=true
export MODEL_NAME="$MODEL_PATH"
export DATASET_NAME="$DATASET_VERSION"

python3 -m orby.trainer.main_eval \
    data.path=$DATA_PATH/$OUTPUT_FILE \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN \
    +custom_reward_function.reward_kwargs.prompt_format=$PROMPT_FORMAT

if [ $? -ne 0 ]; then
    echo "ERROR: Evaluation phase failed"
    exit 1
fi

echo "SUCCESS: Evaluation completed successfully for $DATASET_VERSION with model $MODEL_PATH"
exit 0
