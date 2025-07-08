DATA_PATH="PUT YOUR DATA PATH HERE"
REWARD_FILE=orby/reward/subtask.py
REWARD_FN=train_reward_func
COORDINATES_METRIC=gaussian
COORDINATES_GAUSSIAN_SIGMA=2
COORDINATES_PIXEL_SQUARE_SIZE=10

# Evaluation
python3 -m orby.trainer.main_eval \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.response_key=predictions \
    +data.save_scores=True \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN \
    +custom_reward_function.reward_kwargs.coordinates_metric=$COORDINATES_METRIC \
    +custom_reward_function.reward_kwargs.coordinates_gaussian_sigma=$COORDINATES_GAUSSIAN_SIGMA \
    +custom_reward_function.reward_kwargs.coordinates_pixel_square_size=$COORDINATES_PIXEL_SQUARE_SIZE
