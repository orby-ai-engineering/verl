# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict
import os
import pprint

import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm

from verl.utils.fs import copy_to_local


# Copied from recipe/r1/main_eval.py
def get_custom_reward_fn(config):
    import importlib.util
    import os

    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    if not file_path:
        return None

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    if spec is None:
        raise RuntimeError(f"Could not create module spec for '{file_path}'")
    
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}'") from e

    function_name = reward_fn_config.get("name")
    if function_name is None:
        raise ValueError("Reward function name not specified in config")

    if not hasattr(module, function_name):
        raise AttributeError(
            f"Reward function '{function_name}' not found in '{file_path}'."
        )

    print(f"using customized reward function '{function_name}' from '{file_path}'")

    raw_fn = getattr(module, function_name)

    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


@ray.remote
def process_item(reward_fn, data_source, response_lst, reward_data):
    ground_truth = reward_data["ground_truth"]
    score_lst = [reward_fn(data_source, r, ground_truth) for r in response_lst]
    df = pd.DataFrame(score_lst)

    mean_scores = {}
    for name, score in df.items():
        try:
            mean_scores[name] = np.mean(score)
        except:
            pass

    return data_source, mean_scores


def upload_to_wandb(metric_dict, model_name=None, dataset_name=None):
    """
    Upload evaluation results to Weights & Biases as a table.
    
    Args:
        metric_dict: Dictionary containing evaluation metrics
        model_name: Name of the model being evaluated
        dataset_name: Name of the dataset being evaluated
    """
    try:
        import wandb
        
        # Initialize wandb run
        wandb.init(
            project="eval_screenspot",
            name=f"{model_name}_{dataset_name}" if model_name and dataset_name else "evaluation",
            config={
                "model_name": model_name,
                "dataset_name": dataset_name,
                "evaluation_type": "screenspot"
            }
        )
        
        # Create a table from the metric dictionary
        # Convert the nested dictionary structure to a flat table
        table_data = []
        
        for metric_key, metric_value in metric_dict.items():
            # Parse the metric key to extract components
            # Format: test_score/{data_source}/{metric_name}
            parts = metric_key.split('/')
            if len(parts) >= 3:
                data_source = parts[1]
                metric_name = '/'.join(parts[2:])
                
                table_data.append({
                    "data_source": data_source,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "full_metric_key": metric_key
                })
        
        if table_data:
            # Create wandb table
            table = wandb.Table(columns=["data_source", "metric_name", "metric_value", "full_metric_key"])
            for row in table_data:
                table.add_data(row["data_source"], row["metric_name"], row["metric_value"], row["full_metric_key"])
            
            # Log the table
            wandb.log({"evaluation_results": table})
            
            # Also log individual metrics for easier tracking
            for metric_key, metric_value in metric_dict.items():
                wandb.log({metric_key: float(metric_value)})
            
            print(f"Successfully uploaded {len(table_data)} metrics to wandb")
        else:
            print("Warning: No metrics to upload to wandb")
            
        wandb.finish()
        
    except ImportError:
        print("Warning: wandb not installed. Skipping wandb upload.")
    except Exception as e:
        print(f"Error uploading to wandb: {e}")


@hydra.main(
    config_path="../../verl/trainer/config", config_name="evaluation", version_base=None
)
def main(config):
    local_path = copy_to_local(config.data.path)
    dataset = pd.read_parquet(
        local_path,
        columns=[
            config.data.response_key,
            config.data.data_source_key,
            config.data.reward_model_key,
        ],
    )
    responses = dataset[config.data.response_key]
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)

    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_cpus=config.ray_init.num_cpus)

    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    compute_score = get_custom_reward_fn(config)

    # Create remote tasks
    remote_tasks = [
        process_item.remote(
            compute_score, data_sources[i], responses[i], reward_model_data[i]
        )
        for i in range(total)
    ]

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                data_source, mean_scores = ray.get(result_id)
                data_source_reward[data_source].append(mean_scores)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        rewards = pd.DataFrame(rewards)
        rewards = rewards.mean()
        for k, v in rewards.items():
            metric_dict[f"test_score/{data_source}/{k}"] = v

    pprint.pprint(metric_dict)
    
    # Upload to wandb if environment variable is set
    if os.getenv("UPLOAD_TO_WANDB", "false").lower() == "true":
        model_name = os.getenv("MODEL_NAME")
        dataset_name = os.getenv("DATASET_NAME")
        upload_to_wandb(metric_dict, model_name, dataset_name)


if __name__ == "__main__":
    main()
