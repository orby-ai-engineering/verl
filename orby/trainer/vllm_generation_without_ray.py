import os
import json
import torch
from typing import List, Dict, Any
from openai import OpenAI
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import base64
from io import BytesIO

@dataclass
class Batch:
    """Class to hold a batch of data"""
    prompts: List[str]
    ground_truth: List[Dict[str, Any]]

class DataLoader:
    """Data loader for loading and batching data"""
    def __init__(self, data_path: str, batch_size: int = 128):
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.data = self._load_data()
        
    def _load_data(self) -> pd.DataFrame:
        """Load data from the parquet file"""
        return pd.read_parquet(self.data_path)
    
    def _create_chat_messages(self, row: pd.Series) -> str:
        """Create chat template for the prompt"""
        prompt = row['prompt']

        # Convert image bytes to base64 encoding
        if len(row['images']) != 1:
            raise ValueError("More than 1 image")
        image_bytes = BytesIO(row['images'][0]['bytes']).getvalue()
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[0]["content"]
                    }
                ]    
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt[1]["content"]
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        return messages
    
    def get_batches(self):
        """Generate batches of data"""
        for i in range(0, len(self.data), self.batch_size):
            batch_data = self.data.iloc[i:i + self.batch_size]
            prompts = [self._create_chat_messages(row) for _, row in batch_data.iterrows()]
            ground_truth = batch_data['reward_model'].tolist()
            yield Batch(prompts=prompts, ground_truth=ground_truth)

class VLLMClient:
    """Client for interacting with VLLM server"""
    def __init__(self, server_url: str):
        self.client = OpenAI(
            base_url=server_url,
            api_key="not-needed"  # VLLM server doesn't require API key
        )
        
    def generate(self, prompt: Any) -> str:
        """Generate response for a single prompt"""
        completion = self.client.chat.completions.create(
            model="qwen25vl7b-2",  # Model name not needed as it's configured on server
            messages=prompt,
            temperature=0,
            # max_tokens=2048,
            top_p=1.0,
            # frequency_penalty=0.0,
            # presence_penalty=0.0
        )
        return completion.choices[0].message.content

def process_batch(batch: Batch, client: VLLMClient) -> List[str]:
    """Process a batch of prompts using a single client"""
    responses = []
    for prompt in batch.prompts:
        response = client.generate(prompt)
        responses.append([response])
    return responses

def main():
    # Configuration
    data_path = "~/data/screenspot/test.parquet"  # Replace with your data path
    server_url = "http://model.orbyapi.com/v1"
    batch_size = 16
    output_file = os.path.join(os.path.dirname(data_path), "responses.parquet")
    
    # Create a single VLLM client
    client = VLLMClient(server_url)

    # Initialize components
    data_loader = DataLoader(data_path, batch_size)
    
    # Process batches and collect responses
    all_responses = []
    
    for batch in tqdm(data_loader.get_batches(), desc="Processing batches"):
        # Generate responses from VLLM server sequentially
        responses = process_batch(batch, client)
        all_responses.extend(responses)
   
    # Add responses to the original dataset
    data_loader.data['responses'] = all_responses
    
    # Save the updated dataset
    data_loader.data.to_parquet(output_file, index=False)
    print(f"Saved dataset with {len(all_responses)} responses to {output_file}")

if __name__ == "__main__":
    main() 