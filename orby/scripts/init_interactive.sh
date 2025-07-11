YOUR_EMAIL=""
YOUR_NAME=""
if [ -z "$YOUR_EMAIL" ] || [ -z "$YOUR_NAME" ]; then
    echo "Please set YOUR_EMAIL and YOUR_NAME in the script"
    exit 1
fi
echo Using $YOUR_EMAIL and $YOUR_NAME for git config
git config --global user.email $YOUR_EMAIL
git config --global user.name $YOUR_NAME

# Accelerate package download speed
sed -i 's|mirrors.tuna.tsinghua.edu.cn|archive.ubuntu.com|g' /etc/apt/sources.list

apt update
apt install -y emacs
apt install -y awscli
# urllib3<2 required by awscli
pip install 'urllib3<2'
pip install parquet-tools

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source $HOME/miniconda/bin/activate

# Create and setup conda environment
conda create -n verl python=3.12 -y
conda activate verl

# Clone VERL repository
git clone git@github.com:orby-ai-engineering/verl.git && cd verl
# Initialize git submodules if needed
git submodule update --init --recursive
# Install dependencies
pip install -e .[vllm]

# Handle Flash Attention installation
pip uninstall -y flash-attn
FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn --no-build-isolation

# Install additional utilities
pip install qwen_vl_utils
pip install qwen_agent
pip install hf_transfer

# Download model, verify transformers installation
python3 -c "import transformers; transformers.pipeline(model='Qwen/Qwen2.5-VL-7B-Instruct')"

# Install internal packages
conda deactivate
conda create -n digital-agent python=3.12 -y
git clone git@github.com:orby-ai-engineering/digital-agent.git & cd digital-agent
git submodule update --init --recursive
pip install -r requirements.txt


# Download and convert action description dev set
# mkdir -p ~/data/action_description/raw/
# aws s3 cp s3://orby-osu-va/mds_datasets/Q42024_Intake_Format/ActIO-ActionDescription/parquet/dev.parquet ~/data/action_description/raw/dev.parquet
# python orby/data/convert_action_description.py --input_file=~/data/action_description/raw/dev.parquet --split=train
# python orby/data/convert_action_description.py --input_file=~/data/action_description/raw/dev.parquet --split=test

# Download the subtask direct distill dataset
# # Separate E-RM datasets
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/executor.parquet ~/data/subtask_direct_distill/mix/test/executor.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/reward_model.parquet ~/data/subtask_direct_distill/mix/test/reward_model.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/executor_block512mb/part-00000-tid-8818964477925489584-78825fd1-c751-4bef-9f26-0c77bbdb2020-258-1-c000.snappy.parquet ~/data/subtask_direct_distill/mix/train/executor.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/reward_model_block512mb/part-00000-tid-8036795855418196458-5aac9d57-8404-4b87-b9f1-088ad4820f8f-136-1-c000.snappy.parquet ~/data/subtask_direct_distill/mix/train/reward_model.parquet
# # Merged dataset
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/executor_reward_model_combined.parquet ~/data/subtask_direct_distill/mix/test/combined.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/executor_reward_model_combined_block512mb/part-00000-tid-3008114883591573361-eb7554a3-5626-4452-8b82-4ba9fa62e452-352-1-c000.snappy.parquet ~/data/subtask_direct_distill/mix/train/combined.parquet
# # Merged dataset with response
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/test/executor_reward_model_combined_with_response/part-00000-tid-6116228279497623000-6660e492-d87d-4c87-903b-261c86c79b92-531-1-c000.snappy.parquet ~/data/subtask_direct_distill/mix/test/combined_with_response.parquet
# aws s3 cp s3://orby-osu-va/subtask/verl/experiment_2/train/executor_reward_model_combined_with_response/part-00000-tid-3712964276653840281-af2210b2-e910-4427-aa16-9f2a2cfdae0a-844-1-c000.snappy.parquet ~/data/subtask_direct_distill/mix/train/combined_with_response.parquet
