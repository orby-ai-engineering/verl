MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
MODEL_PATH=checkpoints/verl_grpo_example_subtask_hsmv2_distill/qwen2_5_vl_7b_subtask_hsmv2_distill/global_step_440/merged_model
#DATA_PATH=/workspace/datasets/subtask/claude35_normal_cc_train_5k/executor_dataset/verl_training_format/chunk_047.parquet
#DATA_PATH=/workspace/datasets/subtask/claude35_normal_cc_train_5k/executor_dataset/all_verl_data.parquet
#DATA_PATH=/workspace/datasets/cheng_dataset/executor_0060.parquet
DATA_PATH=/workspace/datasets/subtask/claude35_normal_cc_train_5k/executor_dataset/test_verl_data.parquet
REWARD_FILE=orby/reward/subtask.py
REWARD_FN=reward_func
OUTPUT_FILE=/workspace/datasets/subtask/claude35_normal_cc_train_5k/executor_dataset/eval_results/results_trained_model.parquet
# OUTPUT_FILE=/workspace/datasets/subtask/claude35_normal_cc_train_5k/executor_dataset/eval_results/results_1.parquet

# Generation
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Generation
python3 -m orby.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=$DATA_PATH \
    data.prompt_key=prompt \
    data.batch_size=1024 \
    +data.max_prompt_length=7680 \
    data.n_samples=1 \
    data.output_path=$OUTPUT_FILE \
    model.path=$MODEL_PATH \
    rollout.prompt_length=7680 \
    rollout.response_length=512 \
    rollout.tensor_model_parallel_size=1 \
    rollout.gpu_memory_utilization=0.9 \
    rollout.max_num_batched_tokens=65536 \
    +rollout.limit_images=3

# Evaluation
python3 -m orby.trainer.main_eval \
    data.path=$OUTPUT_FILE \
    data.prompt_key=prompt \
    data.response_key=responses \
    custom_reward_function.path=$REWARD_FILE \
    custom_reward_function.name=$REWARD_FN
