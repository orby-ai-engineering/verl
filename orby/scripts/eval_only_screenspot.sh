set -e

# Default values
DATASET_VERSION="screenspot"
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
REWARD_FILE=orby/reward/screenspot.py
REWARD_FN=reward_func
OUTPUT_FILE=responses.parquet
PROMPT_FORMAT="qwen"
DATA_PATH=~/data/$DATASET_VERSION/

echo "Using dataset version: $DATASET_VERSION"
echo "Data path: $DATA_PATH"

# Evaluation
python3 -m orby.trainer.main_eval \
    data.path=$DATA_PATH/$OUTPUT_FILE \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN
