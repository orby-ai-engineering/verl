"""
Unit tests for main_reward_filter module
"""

import ast
import os
import tempfile
import shutil
from unittest.mock import patch

import pandas as pd
import pytest

from orby.trainer.main_reward_filter import (
    parse_filter_bounds,
    filter_parquet_chunks,
)


class TestRewardFilter:
    """Test suite for reward filtering functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample data for testing with proper nested structure
        self.sample_data = pd.DataFrame({
            'prompt': ['prompt1', 'prompt2', 'prompt3', 'prompt4', 'prompt5'],
            'response': ['response1', 'response2', 'response3', 'response4', 'response5'],
            'reward_score': [0.3, 0.7, 0.9, 0.4, 0.8],
            'reward_model': [
                {'ground_truth': {'should_end': 'true'}},
                {'ground_truth': {'should_end': 'false'}},
                {'ground_truth': {'should_end': 'true'}},
                {'ground_truth': {'should_end': 'false'}},
                {'ground_truth': {'should_end': 'true'}}
            ],
            'other_column': ['a', 'b', 'c', 'd', 'e']
        })
        
        # Create input parquet file
        self.input_path = os.path.join(self.test_dir, 'input.parquet')
        self.sample_data.to_parquet(self.input_path, index=False)
        
        # Set up output paths
        self.output_path = os.path.join(self.test_dir, 'output.parquet')

    def teardown_method(self):
        """Clean up test fixtures after each test method."""
        shutil.rmtree(self.test_dir)

    def test_parse_filter_bounds_valid_input(self):
        """Test parsing valid filter bounds string."""
        bounds_str = "[0.51, 0.9]"
        result = parse_filter_bounds(bounds_str)
        assert result == (0.51, 0.9)

    def test_parse_filter_bounds_integer_values(self):
        """Test parsing filter bounds with integer values."""
        bounds_str = "[0, 1]"
        result = parse_filter_bounds(bounds_str)
        assert result == (0, 1)

    def test_parse_filter_bounds_negative_values(self):
        """Test parsing filter bounds with negative values."""
        bounds_str = "[-0.5, 0.5]"
        result = parse_filter_bounds(bounds_str)
        assert result == (-0.5, 0.5)

    def test_parse_filter_bounds_invalid_format(self):
        """Test parsing invalid filter bounds format."""
        with pytest.raises(ValueError):
            parse_filter_bounds("invalid")

    def test_filter_parquet_chunks_basic_filtering(self):
        """Test basic filtering functionality without balancing."""
        filter_bounds = (0.6, 1.0)
        
        rows_kept = filter_parquet_chunks(
            self.input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=False,
            chunk_size=2
        )
        
        # Should keep rows with reward_score >= 0.6: rows 1, 2, 4 (0.7, 0.9, 0.8)
        assert rows_kept == 3
        
        # Check output file
        output_df = pd.read_parquet(self.output_path)
        assert len(output_df) == 3
        assert all(output_df['reward_score'] >= 0.6)

    def test_filter_parquet_chunks_with_balancing(self):
        """Test filtering with should_end balancing."""
        filter_bounds = (0.6, 1.0)
        
        rows_kept = filter_parquet_chunks(
            self.input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=True,
            should_end_column='reward_model.ground_truth.should_end',
            chunk_size=2
        )
        
        # Original filtered data: rows 1, 2, 4 (0.7, 0.9, 0.8)
        # should_end values: false, true, true
        # should_end_true_count = 2, should_end_false_count = 1
        # Need to add 1 more should_end=false row
        # Available balancing data: rows 0, 3 (both have should_end=false and don't meet filter)
        assert rows_kept == 4  # 3 original + 1 balancing

        output_df = pd.read_parquet(self.output_path)
        assert len(output_df) == 4
        
        # Check balancing worked
        from orby.trainer.main_reward_filter import extract_should_end_values
        should_end_values = extract_should_end_values(output_df, 'reward_model.ground_truth.should_end')
        should_end_counts = pd.Series(should_end_values).value_counts()
        assert should_end_counts['true'] == should_end_counts['false']

    def test_filter_parquet_chunks_no_balancing_needed(self):
        """Test filtering when no balancing is needed."""
        # Create data where should_end is already balanced
        balanced_data = pd.DataFrame({
            'prompt': ['prompt1', 'prompt2', 'prompt3', 'prompt4'],
            'response': ['response1', 'response2', 'response3', 'response4'],
            'reward_score': [0.7, 0.8, 0.9, 0.6],
            'reward_model': [
                {'ground_truth': {'should_end': 'true'}},
                {'ground_truth': {'should_end': 'false'}},
                {'ground_truth': {'should_end': 'true'}},
                {'ground_truth': {'should_end': 'false'}}
            ]
        })
        
        balanced_input_path = os.path.join(self.test_dir, 'balanced_input.parquet')
        balanced_data.to_parquet(balanced_input_path, index=False)
        
        filter_bounds = (0.6, 1.0)
        
        rows_kept = filter_parquet_chunks(
            balanced_input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=True,
            should_end_column='reward_model.ground_truth.should_end',
            chunk_size=2
        )
        
        # All 4 rows should be kept, no balancing needed
        assert rows_kept == 4

    def test_filter_parquet_chunks_missing_reward_column(self):
        """Test error when reward score column is missing."""
        # Create data without reward_score column
        data_without_reward = pd.DataFrame({
            'prompt': ['prompt1', 'prompt2'],
            'response': ['response1', 'response2'],
            'other_column': ['a', 'b']
        })
        
        input_path_no_reward = os.path.join(self.test_dir, 'no_reward.parquet')
        data_without_reward.to_parquet(input_path_no_reward, index=False)
        
        filter_bounds = (0.6, 1.0)
        
        with pytest.raises(ValueError, match="No 'reward_score' column found"):
            filter_parquet_chunks(
                input_path_no_reward,
                self.output_path,
                filter_bounds,
                balance_should_end=False
            )

    def test_filter_parquet_chunks_custom_reward_column(self):
        """Test filtering with custom reward score column name."""
        # Create data with custom reward column
        custom_data = pd.DataFrame({
            'prompt': ['prompt1', 'prompt2', 'prompt3'],
            'response': ['response1', 'response2', 'response3'],
            'custom_reward': [0.3, 0.7, 0.9]
        })
        
        custom_input_path = os.path.join(self.test_dir, 'custom_reward.parquet')
        custom_data.to_parquet(custom_input_path, index=False)
        
        filter_bounds = (0.6, 1.0)
        
        rows_kept = filter_parquet_chunks(
            custom_input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=False,
            reward_score_column='custom_reward'
        )
        
        # Should keep rows with custom_reward >= 0.6: rows 1, 2 (0.7, 0.9)
        assert rows_kept == 2
        
        output_df = pd.read_parquet(self.output_path)
        assert len(output_df) == 2
        assert all(output_df['custom_reward'] >= 0.6)

    def test_filter_parquet_chunks_missing_should_end_column(self):
        """Test handling when should_end column is missing."""
        # Create data without should_end column
        data_without_should_end = pd.DataFrame({
            'prompt': ['prompt1', 'prompt2'],
            'response': ['response1', 'response2'],
            'reward_score': [0.7, 0.8]
        })
        
        input_path_no_should_end = os.path.join(self.test_dir, 'no_should_end.parquet')
        data_without_should_end.to_parquet(input_path_no_should_end, index=False)
        
        filter_bounds = (0.6, 1.0)
        
        # Should not raise error, just skip balancing
        rows_kept = filter_parquet_chunks(
            input_path_no_should_end,
            self.output_path,
            filter_bounds,
            balance_should_end=True,
            should_end_column='reward_model.ground_truth.should_end'
        )
        
        # Should keep both rows
        assert rows_kept == 2

    def test_filter_parquet_chunks_empty_result(self):
        """Test filtering that results in empty output."""
        filter_bounds = (1.5, 2.0)  # No rows should match
        
        rows_kept = filter_parquet_chunks(
            self.input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=False
        )
        
        assert rows_kept == 0
        # Output file should not exist when no rows are kept
        assert not os.path.exists(self.output_path)

    def test_filter_parquet_chunks_insufficient_balancing_data(self):
        """Test balancing when there's insufficient balancing data."""
        # Create data where we need more balancing data than available
        imbalanced_data = pd.DataFrame({
            'prompt': ['p1', 'p2', 'p3', 'p4'],
            'response': ['r1', 'r2', 'r3', 'r4'],
            'reward_score': [0.8, 0.9, 0.7, 0.2],  # Only row 3 available for balancing
            'reward_model': [
                {'ground_truth': {'should_end': 'true'}},
                {'ground_truth': {'should_end': 'true'}},
                {'ground_truth': {'should_end': 'true'}},
                {'ground_truth': {'should_end': 'false'}}
            ]
        })
        
        imbalanced_input_path = os.path.join(self.test_dir, 'imbalanced_input.parquet')
        imbalanced_data.to_parquet(imbalanced_input_path, index=False)
        
        filter_bounds = (0.6, 1.0)
        
        rows_kept = filter_parquet_chunks(
            imbalanced_input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=True,
            should_end_column='reward_model.ground_truth.should_end'
        )
        
        # Should keep 3 filtered rows + 1 balancing row (all available)
        assert rows_kept == 4
        
        output_df = pd.read_parquet(self.output_path)
        from orby.trainer.main_reward_filter import extract_should_end_values
        should_end_values = extract_should_end_values(output_df, 'reward_model.ground_truth.should_end')
        should_end_counts = pd.Series(should_end_values).value_counts()
        assert should_end_counts['true'] == 3
        assert should_end_counts['false'] == 1

    def test_filter_parquet_chunks_output_directory_creation(self):
        """Test that output directory is created if it doesn't exist."""
        nested_output_path = os.path.join(self.test_dir, 'nested', 'subdir', 'output.parquet')
        
        filter_bounds = (0.6, 1.0)
        
        rows_kept = filter_parquet_chunks(
            self.input_path,
            nested_output_path,
            filter_bounds,
            balance_should_end=False
        )
        
        assert rows_kept == 3
        assert os.path.exists(nested_output_path)

    def test_filter_parquet_chunks_preserves_all_columns(self):
        """Test that all columns are preserved in the output."""
        filter_bounds = (0.6, 1.0)
        
        filter_parquet_chunks(
            self.input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=False
        )
        
        output_df = pd.read_parquet(self.output_path)
        input_df = pd.read_parquet(self.input_path)
        
        # All columns should be preserved
        assert list(output_df.columns) == list(input_df.columns)

    def test_filter_parquet_chunks_small_chunk_size(self):
        """Test filtering with very small chunk size."""
        filter_bounds = (0.6, 1.0)
        
        rows_kept = filter_parquet_chunks(
            self.input_path,
            self.output_path,
            filter_bounds,
            balance_should_end=False,
            chunk_size=1  # Process one row at a time
        )
        
        assert rows_kept == 3
        
        output_df = pd.read_parquet(self.output_path)
        assert len(output_df) == 3
        assert all(output_df['reward_score'] >= 0.6)

    @patch('orby.trainer.main_reward_filter.copy_to_local')
    def test_main_function_integration(self, mock_copy_to_local):
        """Test the main function integration with hydra config."""
        from orby.trainer.main_reward_filter import main
        from omegaconf import DictConfig
        
        # Mock copy_to_local to return our test input path
        mock_copy_to_local.return_value = self.input_path
        
        # Create config
        config = DictConfig({
            'data': {
                'path': '/mock/path/input.parquet',
                'medium_difficulty_output_path': os.path.join(self.test_dir, 'medium.parquet'),
                'hard_difficulty_output_path': os.path.join(self.test_dir, 'hard.parquet'),
                'medium_difficulty_filter_bound': '[0.7, 1.0]',
                'hard_difficulty_filter_bound': '[0.0, 0.5]',
                'balance_should_end': True,
                'should_end_column': 'reward_model.ground_truth.should_end',
                'reward_score_column': 'reward_score'
            }
        })
        
        # Run main function
        main(config)
        
        # Check that both output files were created
        assert os.path.exists(os.path.join(self.test_dir, 'medium.parquet'))
        assert os.path.exists(os.path.join(self.test_dir, 'hard.parquet'))
        
        # Check medium difficulty output
        medium_df = pd.read_parquet(os.path.join(self.test_dir, 'medium.parquet'))
        # Note: Due to balancing, some rows might have reward_score < 0.7
        # Check that original filtered rows meet the criteria
        original_filtered = medium_df[medium_df['reward_score'] >= 0.7]
        assert len(original_filtered) >= 2  # Should have at least 2 rows meeting criteria
        
        # Check hard difficulty output
        hard_df = pd.read_parquet(os.path.join(self.test_dir, 'hard.parquet'))
        # Check that original filtered rows meet the criteria
        original_filtered_hard = hard_df[hard_df['reward_score'] <= 0.5]
        assert len(original_filtered_hard) >= 2  # Should have at least 2 rows meeting criteria 