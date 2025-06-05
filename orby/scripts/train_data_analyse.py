import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from collections import Counter

def analyze_training_data(parquet_file_path):
    """Analyze and visualize SFT training data structure"""
    
    # Load the training data
    df = pd.read_parquet(parquet_file_path)
    
    print("=== TRAINING DATA BASIC INFO ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")
    
    print("\n=== SAMPLE ROWS ===")
    for i in range(min(15, len(df))):
        print(f"\nRow {i}:")
        for col in df.columns:
            value = df.iloc[i][col]
            if isinstance(value, (list, np.ndarray)):
                print(f"  {col}: {type(value)} with {len(value)} items")
                if len(value) > 0:
                    print(f"    First item: {str(value[0])[:200]}...")
            else:
                print(f"  {col}: {type(value)} - {str(value)[:200]}...")
    
    # Analyze conversation structure
    print("\n=== CONVERSATION STRUCTURE ANALYSIS ===")
    
    if 'conversations' in df.columns:
        analyze_conversations(df['conversations'])
    elif 'prompt' in df.columns and 'response' in df.columns:
        analyze_prompt_response(df)
    elif 'messages' in df.columns:
        analyze_messages(df['messages'])
    else:
        print("Unknown conversation format. Available columns:", df.columns.tolist())
        # Try to analyze any column that might contain conversation data
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['conv', 'chat', 'dialog', 'message', 'prompt', 'response']):
                print(f"\nAnalyzing column: {col}")
                analyze_generic_column(df[col])
    
    # Analyze coordinate patterns if present
    analyze_coordinate_patterns(df)
    
    # Create visualizations
    create_training_data_visualizations(df)

def analyze_conversations(conversations_col):
    """Analyze conversations column structure"""
    print("Analyzing 'conversations' column...")
    
    conversation_lengths = []
    role_counts = Counter()
    has_coordinates = 0
    has_think_answer = 0
    empty_responses = 0
    
    for i, conv in enumerate(conversations_col):
        if isinstance(conv, (list, np.ndarray)):
            conversation_lengths.append(len(conv))
            
            for turn in conv:
                if isinstance(turn, dict):
                    role = turn.get('role', 'unknown')
                    content = turn.get('content', '')
                    role_counts[role] += 1
                    
                    # Check for coordinate patterns
                    if any(pattern in str(content).lower() for pattern in ['<answer>', 'coordinates', 'click']):
                        has_coordinates += 1
                    
                    # Check for think/answer format
                    if '<think>' in str(content) and '<answer>' in str(content):
                        has_think_answer += 1
                    
                    # Check for empty responses
                    if role == 'assistant' and len(str(content).strip()) == 0:
                        empty_responses += 1
                        
            # Show first few examples
            if i < 5:
                print(f"\nConversation {i}:")
                for turn in conv:
                    if isinstance(turn, dict):
                        role = turn.get('role', 'unknown')
                        content = str(turn.get('content', ''))[:150]
                        print(f"  {role}: {content}...")
    
    print(f"\nConversation Statistics:")
    print(f"  Average conversation length: {np.mean(conversation_lengths):.1f} turns")
    print(f"  Role distribution: {dict(role_counts)}")
    print(f"  Conversations with coordinates: {has_coordinates}")
    print(f"  Conversations with <think><answer> format: {has_think_answer}")
    print(f"  Empty assistant responses: {empty_responses}")

def analyze_prompt_response(df):
    """Analyze prompt-response format"""
    print("Analyzing prompt-response format...")
    
    prompt_lengths = []
    response_lengths = []
    has_responses = 0
    empty_responses = 0
    
    for i, row in df.iterrows():
        prompt = str(row.get('prompt', ''))
        response = str(row.get('response', ''))
        
        prompt_lengths.append(len(prompt.split()))
        response_lengths.append(len(response.split()))
        
        if len(response.strip()) > 0:
            has_responses += 1
        else:
            empty_responses += 1
            
        # Show first few examples
        if i < 5:
            print(f"\nExample {i}:")
            print(f"  Prompt: {prompt[:200]}...")
            print(f"  Response: {response[:200]}...")
    
    print(f"\nPrompt-Response Statistics:")
    print(f"  Average prompt length: {np.mean(prompt_lengths):.1f} words")
    print(f"  Average response length: {np.mean(response_lengths):.1f} words")
    print(f"  Samples with responses: {has_responses}/{len(df)}")
    print(f"  Empty responses: {empty_responses}")

def analyze_messages(messages_col):
    """Analyze messages column structure"""
    print("Analyzing 'messages' column...")
    # Similar to conversations but for messages format
    analyze_conversations(messages_col)  # Reuse the same logic

def analyze_generic_column(column):
    """Analyze any column that might contain conversation data"""
    print(f"Sample values from this column:")
    for i, value in enumerate(column.head(3)):
        print(f"  Row {i}: {type(value)} - {str(value)[:300]}...")

def analyze_coordinate_patterns(df):
    """Look for coordinate patterns in the data"""
    print("\n=== COORDINATE PATTERN ANALYSIS ===")
    
    coordinate_patterns = [
        r'\d+\s+\d+',  # "123 456"
        r'\(\d+,\s*\d+\)',  # "(123, 456)"
        r'<answer>\d+\s+\d+</answer>',  # "<answer>123 456</answer>"
        r'x:\s*\d+.*y:\s*\d+',  # "x: 123, y: 456"
    ]
    
    pattern_counts = {pattern: 0 for pattern in coordinate_patterns}
    
    # Check all text columns for coordinate patterns
    for col in df.columns:
        if df[col].dtype == 'object':
            for value in df[col]:
                text = str(value).lower()
                for pattern in coordinate_patterns:
                    import re
                    if re.search(pattern, text):
                        pattern_counts[pattern] += 1
    
    print("Coordinate pattern matches:")
    for pattern, count in pattern_counts.items():
        print(f"  {pattern}: {count} matches")

def create_training_data_visualizations(df):
    """Create visualizations for training data"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Data size distribution
    axes[0, 0].bar(['Total Samples'], [len(df)])
    axes[0, 0].set_title('Training Data Size')
    axes[0, 0].set_ylabel('Number of Samples')
    
    # 2. Column types
    col_types = df.dtypes.value_counts()
    axes[0, 1].pie(col_types.values, labels=col_types.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Column Data Types')
    
    # 3. Text length distribution (if applicable)
    text_lengths = []
    for col in df.columns:
        if df[col].dtype == 'object':
            lengths = [len(str(x)) for x in df[col].head(100)]  # Sample first 100
            text_lengths.extend(lengths)
    
    if text_lengths:
        axes[1, 0].hist(text_lengths, bins=30, alpha=0.7)
        axes[1, 0].set_title('Text Length Distribution')
        axes[1, 0].set_xlabel('Character Count')
        axes[1, 0].set_ylabel('Frequency')
    
    # 4. Missing data analysis
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        missing_data.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Missing Data by Column')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'No Missing Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('Missing Data Analysis')
    
    plt.tight_layout()
    plt.savefig('training_data_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Training data visualization saved as 'training_data_analysis.png'")

# Usage
if __name__ == "__main__":
    # Replace with your training data path
    training_file = "/root/data/uground/train.parquet"
    
    try:
        analyze_training_data(training_file)
    except FileNotFoundError:
        print(f"File not found: {training_file}")
        print("Please provide the correct path to your training data file.")
    except Exception as e:
        print(f"Error analyzing training data: {e}")
        print("Please check the file format and structure.")
