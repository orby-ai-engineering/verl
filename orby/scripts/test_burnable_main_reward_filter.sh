python3 -m orby.trainer.main_reward_filter \
    data.path=data/test_burnable_data.parquet \
    data.medium_difficulty_output_path=data/test_burnable_data_medium.parquet \
    data.hard_difficulty_output_path=data/test_burnable_data_hard.parquet \
    data.medium_difficulty_filter_bound="[0.51, 0.9]" \
    data.hard_difficulty_filter_bound="[0.09, 0.5]" \
    data.balance_should_end=true \
    data.should_end_column="reward_model.ground_truth.should_end" \
    data.reward_score_column="reward_score"
