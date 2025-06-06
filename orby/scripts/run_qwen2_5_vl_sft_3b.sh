set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2_5_vl_sft_3b.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

# Changes to run orby.trainer.fsdp_sft_trainer
cd "$(dirname "$(dirname "$(dirname "$(realpath "$0")")")")"
export PYTHONPATH="$PWD:$PYTHONPATH"


#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#unset PYTORCH_CUDA_ALLOC_CONF

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m orby.trainer.fsdp_sft_trainer \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=8 \
    data.train_files=$HOME/data/geo3k/train.parquet \
    data.val_files=$HOME/data/geo3k/test.parquet \
    data.prompt_key=prompt \
    data.response_key=extra_info \
    +data.image_key=images \
    +processor.use_fast=true \
    +processor.trust_remote_code=true \
    optim.lr=1e-4 \
    data.response_dict_keys=['answer'] \
    model.partial_pretrain=Qwen/Qwen2.5-VL-7B-Instruct \
    model.fsdp_config.cpu_offload=true \
    model.enable_gradient_checkpointing=true \
    +model.enable_activation_offload=true \
    model.fsdp_config.offload_params=true \
    +model.fsdp_config.param_offload=true \
    +name_or_path=Qwen/Qwen2.5-VL-7B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=uground-sft \
    trainer.experiment_name=uground-sft-qwen-2.5-7b \
    trainer.logger=[console,wandb] \
    trainer.total_training_steps=500 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=false \
    +model.fsdp_config.reshard_after_forward=true \
    +model.use_remove_padding=true \
    model.fsdp_config.wrap_policy.min_num_params=1000000 \
    +model.fsdp_config.optimizer_offload=true
    +model.tensor_parallel_size=8 \
    +model.vocab_parallel=true
