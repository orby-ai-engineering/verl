set -e

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
REWARD_FILE=orby/reward/screenspot.py
REWARD_FN=reward_func
OUTPUT_FILE=result-test-output-1.parquet

# Convert the dataset to parquet format
DATA_PATH=~/data/screenspot
python3 -m orby.data.convert_screenspot

# Convert screenspot v2 to parquet format
# DATA_PATH=~/data/screenspot_v2
# huggingface-cli download OS-Copilot/ScreenSpot-v2 --repo-type dataset --local-dir=$DATA_PATH
# cd $HOME/data/screenspot_v2
# unzip screenspotv2_image.zip
# cd -
# python orby/data/convert_screenspot_v2.py --image_dir=$HOME/data/screenspot_v2/screenspotv2_image/

# Convert screenspot pro to parquet format
# DATA_PATH=~/data/screenspot_pro
# huggingface-cli download likaixin/ScreenSpot-Pro --repo-type dataset --local-dir=$DATA_PATH
# python orby/data/convert_screenspot_pro.py


# Generation
# Screenspot pro has example with more than 16k tokens.
python3 -m orby.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH/test*.parquet \
    data.prompt_key=prompt \
    data.batch_size=256 \
    +data.max_prompt_length=20000 \
    +data.image_key=images \
    data.n_samples=1 \
    data.output_path=$DATA_PATH/$OUTPUT_FILE \
    model.path=$MODEL_PATH \
    rollout.temperature=0 \
    rollout.top_p=1.0 \
    rollout.prompt_length=20000 \
    rollout.response_length=256 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536

# Evaluation
python3 -m orby.trainer.main_eval \
    data.path=$DATA_PATH/$OUTPUT_FILE \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN
