YOUR_EMAIL=""
YOUR_NAME=""
if [ -z "$YOUR_EMAIL" ] || [ -z "$YOUR_NAME" ]; then
    echo "Please set YOUR_EMAIL and YOUR_NAME in the script"
    exit 1
fi
echo Using $YOUR_EMAIL and $YOUR_NAME for git config
git config --global user.email $YOUR_EMAIL
git config --global user.name $YOUR_NAME

apt update
apt install -y emacs
apt install -y awscli
# urllib3<2 required by awscli
pip install 'urllib3<2'
pip install parquet-tools

# Download model.
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

# Install verl lib: https://verl.readthedocs.io/en/latest/start/install.html
pip3 install -e .[vllm]

# Use the following commands as needed

# Download and convert uground dev set
# mkdir -p ~/data/uground/raw/
# aws s3 cp s3://orby-osu-va/mds_datasets/Q42024_Intake_Format/ActIO-ActionDescription/parquet/dev.parquet ~/data/uground/raw/dev.parquet
# python examples/data_preprocess/uground.py --input_file=~/data/uground/raw/dev.parquet --split=train
# python examples/data_preprocess/uground.py --input_file=~/data/uground/raw/dev.parquet --split=test

# Download the subtask direct distill dataset
# mkdir -p ~/data/subtask_direct_distill/mix/train/
# mkdir -p ~/data/subtask_direct_distill/mix/test/
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/executor.parquet ~/data/subtask_direct_distill/mix/test/executor.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/reward_model.parquet ~/data/subtask_direct_distill/mix/test/reward_model.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/executor_block512mb/part-00000-tid-8818964477925489584-78825fd1-c751-4bef-9f26-0c77bbdb2020-258-1-c000.snappy.parquet ~/data/subtask_direct_distill/mix/train/executor.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/reward_model_block512mb/part-00000-tid-8036795855418196458-5aac9d57-8404-4b87-b9f1-088ad4820f8f-136-1-c000.snappy.parquet ~/data/subtask_direct_distill/mix/train/reward_model.parquet
