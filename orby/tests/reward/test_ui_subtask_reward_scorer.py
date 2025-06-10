"""
Unit tests for UISubtaskRewardScorer
"""

from unittest.mock import patch

import numpy as np
import pytest

from orby.reward.subtask import UISubtaskRewardScorer, compute_score


class TestUISubtaskRewardScorer:
    """Test suite for UISubtaskRewardScorer class."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.scorer = UISubtaskRewardScorer()

    def test_init(self):
        """Test that the scorer initializes with correct weights and tags."""
        assert self.scorer.reward_model_tags == [
            "reasoning",
            "should_end",
            "goal_achieved",
            "answer",
        ]
        assert self.scorer.executor_tags == ["thinking", "action"]

    def test_check_text_similarity_identical(self):
        """Test text similarity with identical strings."""
        assert self.scorer._check_text_similarity("hello world", "hello world") is True

    def test_check_text_similarity_case_insensitive(self):
        """Test text similarity is case insensitive."""
        assert self.scorer._check_text_similarity("Hello World", "hello world") is True

    def test_check_text_similarity_high_similarity(self):
        """Test text similarity with high similarity strings."""
        assert self.scorer._check_text_similarity("hello world", "hello world!") is True

    def test_check_text_similarity_low_similarity(self):
        """Test text similarity with low similarity strings."""
        assert (
            self.scorer._check_text_similarity("hello world", "completely different")
            is False
        )

    def test_check_text_similarity_custom_threshold(self):
        """Test text similarity with custom threshold."""
        assert (
            self.scorer._check_text_similarity("hello", "helo", threshold=0.6) is True
        )
        assert (
            self.scorer._check_text_similarity("hello", "xyz", threshold=0.6) is False
        )

    def test_score_reward_model_perfect_match(self):
        """Test reward model scoring with perfect match."""
        prediction = "<reasoning>This is the reasoning</reasoning><should_end>true</should_end><goal_achieved>true</goal_achieved><answer>The answer</answer>"
        ground_truth = {
            "reasoning": "This is the reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "The answer",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth, detailed=True)

        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["reward_model/should_end"] == 1
        assert result["reward_model/goal_achieved"] == 1
        assert result["reward_model/answer"] == 1

    def test_score_reward_model_should_end_false(self):
        """Test reward model scoring when should_end is false."""
        prediction = "<reasoning>Still working on it</reasoning><should_end>false</should_end><goal_achieved>false</goal_achieved><answer>No answer yet</answer>"
        ground_truth = {
            "reasoning": "Still working on it",
            "should_end": "false",
            "goal_achieved": "false",
            "answer": "Should be ignored",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth, detailed=True)

        # Answer should be 0 because predicted answer ("No answer yet") doesn't match empty string
        # (gt answer is set to empty when should_end is false)
        assert result["reward_model/answer"] == 0

    def test_score_reward_model_missing_fields(self):
        """Test reward model scoring with missing fields."""
        prediction = "<reasoning>This is the reasoning</reasoning><goal_achieved>true</goal_achieved>"
        ground_truth = {
            "reasoning": "This is the reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "The answer",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth, detailed=True)

        # Format score should be 2/4 = 0.5 (2 fields present out of 4)
        assert np.isclose(result["format"], 0.5)

    def test_calculate_coordinates_score_both_none(self):
        """Test coordinates scoring when both are None."""
        score = self.scorer._calculate_coordinates_score(
            None, None, metric="gaussian", gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
        )
        assert np.isclose(score, 1.0)

    def test_calculate_coordinates_score_one_none(self):
        """Test coordinates scoring when one is None."""
        score = self.scorer._calculate_coordinates_score(
            [(10, 20)], None, metric="gaussian", gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
        )
        assert np.isclose(score, 0.0)

        score = self.scorer._calculate_coordinates_score(
            None, [(10, 20)], metric="gaussian", gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
        )
        assert np.isclose(score, 0.0)

    def test_calculate_coordinates_score_identical(self):
        """Test coordinates scoring with identical coordinates."""
        coords = [(10, 20), (30, 40)]
        score = self.scorer._calculate_coordinates_score(
            coords, coords, metric="gaussian", gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
        )
        assert np.isclose(score, 1.0)

    def test_calculate_coordinates_score_close(self):
        """Test coordinates scoring with close coordinates."""
        pred_coords = [(10, 20)]
        gt_coords = [(11, 21)]
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="gaussian", gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
        )

        # Should be high but not 1.0 due to Gaussian similarity (around 0.78)
        assert 0.7 < score < 1.0

    def test_calculate_coordinates_score_far(self):
        """Test coordinates scoring with far coordinates."""
        pred_coords = [(10, 20)]
        gt_coords = [(100, 200)]
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="gaussian", gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
        )

        # Should be very low
        assert score < 0.1

    def test_calculate_coordinates_score_different_lengths(self):
        """Test coordinates scoring with different length coordinate lists."""
        pred_coords = [(10, 20)]
        gt_coords = [(10, 20), (30, 40)]
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="gaussian", gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
        )
        assert np.isclose(score, 0.0)

    def test_calculate_coordinates_score_pixel_square_inside(self):
        """Test pixel square coordinates scoring when prediction is inside the square."""
        pred_coords = [(50, 50)]
        gt_coords = [(50, 50)]  # Same coordinates - should be inside
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="pixel_square", gaussian_sigma=2, pixel_square_size=10, gt_bbox=None
        )
        assert np.isclose(score, 1.0)

        # Test with prediction slightly off but still inside 10x10 square
        pred_coords = [(52, 48)]
        gt_coords = [(50, 50)]
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="pixel_square", gaussian_sigma=2, pixel_square_size=10, gt_bbox=None
        )
        assert np.isclose(score, 1.0)

    def test_calculate_coordinates_score_pixel_square_outside(self):
        """Test pixel square coordinates scoring when prediction is outside the square."""
        pred_coords = [(50, 50)]
        gt_coords = [(60, 60)]  # 10 pixels away - should be outside 10x10 square
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="pixel_square", gaussian_sigma=2, pixel_square_size=10, gt_bbox=None
        )
        assert np.isclose(score, 0.0)

    def test_calculate_coordinates_score_pixel_square_edge_case(self):
        """Test pixel square coordinates scoring at the edge of the square."""
        pred_coords = [(55, 55)]  # Exactly at the edge of 10x10 square centered at (50, 50)
        gt_coords = [(50, 50)]
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="pixel_square", gaussian_sigma=2, pixel_square_size=10, gt_bbox=None
        )
        assert np.isclose(score, 1.0)

        # Just outside the edge
        pred_coords = [(55.1, 55.1)]
        gt_coords = [(50, 50)]
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="pixel_square", gaussian_sigma=2, pixel_square_size=10, gt_bbox=None
        )
        assert np.isclose(score, 0.0)

    def test_pixel_square_score_multiple_coordinates(self):
        """Test pixel square scoring with multiple coordinates."""
        pred_coords = [(50, 50), (100, 100)]
        gt_coords = [(52, 48), (98, 102)]  # Both should be inside their respective squares
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="pixel_square", gaussian_sigma=2, pixel_square_size=10, gt_bbox=None
        )
        assert np.isclose(score, 1.0)

        # One inside, one outside
        pred_coords = [(50, 50), (100, 100)]
        gt_coords = [(52, 48), (90, 90)]  # First inside, second outside 10x10 square
        score = self.scorer._calculate_coordinates_score(
            pred_coords, gt_coords, metric="pixel_square", gaussian_sigma=2, pixel_square_size=10, gt_bbox=None
        )
        assert np.isclose(score, 0.5)  # Average of 1.0 and 0.0

    def test_calculate_pixel_square_score_direct(self):
        """Test the pixel square score calculation function directly."""
        # Test center point
        score = self.scorer._calculate_pixel_square_score((50, 50), (50, 50), pixel_square_size=10)
        assert np.isclose(score, 1.0)

        # Test inside square
        score = self.scorer._calculate_pixel_square_score((52, 48), (50, 50), pixel_square_size=10)
        assert np.isclose(score, 1.0)

        # Test outside square
        score = self.scorer._calculate_pixel_square_score((60, 60), (50, 50), pixel_square_size=10)
        assert np.isclose(score, 0.0)

        # Test edge cases (exactly on boundary)
        score = self.scorer._calculate_pixel_square_score((55, 55), (50, 50), pixel_square_size=10)
        assert np.isclose(score, 1.0)

        score = self.scorer._calculate_pixel_square_score((45, 45), (50, 50), pixel_square_size=10)
        assert np.isclose(score, 1.0)

        # Test with coordinates at origin (test max(coord - size/2, 0) logic)
        score = self.scorer._calculate_pixel_square_score((2, 2), (0, 0), pixel_square_size=10)
        assert np.isclose(score, 1.0)

    def test_calculate_action_args_score_both_none(self):
        """Test action args scoring when both are None."""
        score = self.scorer._calculate_action_args_score(None, None)
        assert np.isclose(score, 1.0)

    def test_calculate_action_args_score_one_none(self):
        """Test action args scoring when one is None."""
        score = self.scorer._calculate_action_args_score({"key": "value"}, None)
        assert np.isclose(score, 0.0)

        score = self.scorer._calculate_action_args_score(None, {"key": "value"})
        assert np.isclose(score, 0.0)

    def test_calculate_action_args_score_key_mismatch(self):
        """Test action args scoring with key mismatch."""
        pred_args = {"key1": "value1"}
        gt_args = {"key2": "value2"}
        score = self.scorer._calculate_action_args_score(pred_args, gt_args)
        assert np.isclose(score, 0.0)

    def test_calculate_action_args_score_identical(self):
        """Test action args scoring with identical args."""
        args = {"button": "left", "double": "false"}
        score = self.scorer._calculate_action_args_score(args, args)
        assert np.isclose(score, 1.0)

    def test_calculate_action_args_score_similar(self):
        """Test action args scoring with similar args."""
        pred_args = {"text": "hello world"}
        gt_args = {"text": "hello world!"}
        score = self.scorer._calculate_action_args_score(pred_args, gt_args)

        # Should be high due to text similarity
        assert score > 0.8

    def test_score_executor_perfect_match(self):
        """Test executor scoring with perfect match."""
        prediction = "<thinking>I need to click the button</thinking><action>click(100, 200, button='left', double=False)</action>"
        ground_truth = {
            "thinking": "I need to click the button",
            "action": "click(100, 200, button='left', double=False)",
        }

        result = self.scorer._score_executor(
            prediction, ground_truth, detailed=True, 
            coordinates_metric="gaussian", coordinates_gaussian_sigma=2, coordinates_pixel_square_size=5
        )

        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["executor/action_type"] == 1
        assert np.isclose(result["executor/coordinates"], 1.0)
        assert np.isclose(result["executor/action_args"], 1.0)

    def test_score_executor_pixel_square_metric(self):
        """Test executor scoring with pixel square metric."""
        prediction = "<thinking>I need to click the button</thinking><action>click(100, 200)</action>"
        ground_truth = {
            "thinking": "I need to click the button",
            "action": "click(102, 198)",  # Should be inside 10x10 square around (100, 200)
        }

        result = self.scorer._score_executor(
            prediction, ground_truth, detailed=True, 
            coordinates_metric="pixel_square", coordinates_gaussian_sigma=2, coordinates_pixel_square_size=10
        )

        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["executor/action_type"] == 1
        assert np.isclose(result["executor/coordinates"], 1.0)

    def test_score_executor_invalid_action(self):
        """Test executor scoring with invalid action."""
        prediction = "<thinking>I need to do something</thinking><action>invalid_action()</action>"
        ground_truth = {
            "thinking": "I need to do something",
            "action": "click(100, 200)",
        }

        with patch("builtins.print"):  # Mock print to avoid output during tests
            result = self.scorer._score_executor(
                prediction, ground_truth, detailed=True,
                coordinates_metric="gaussian", coordinates_gaussian_sigma=2, coordinates_pixel_square_size=5
            )

        # Should only get format score
        assert result["score"] > 0
        assert result["format"] > 0
        assert result["executor/action_type"] == 0
        assert result["executor/coordinates"] == 0
        assert result["executor/action_args"] == 0

    def test_score_reward_model_type(self):
        """Test score method with reward model type ground truth."""
        ground_truth = {
            "reasoning": "Test reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "Test answer",
        }

        prediction = "<reasoning>Test reasoning</reasoning><should_end>true</should_end><goal_achieved>true</goal_achieved><answer>Test answer</answer>"

        result = self.scorer.score(prediction, ground_truth)
        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["reward_model/should_end"] == 1
        assert result["reward_model/goal_achieved"] == 1
        assert result["reward_model/answer"] == 1

    def test_score_executor_type(self):
        """Test score method with executor type ground truth."""
        ground_truth = {"thinking": "Test thinking", "action": "click(100, 200)"}

        prediction = (
            "<thinking>Test thinking</thinking><action>click(100, 200)</action>"
        )

        result = self.scorer.score(prediction, ground_truth)
        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["executor/action_type"] == 1
        assert np.isclose(result["executor/coordinates"], 1.0)
        assert np.isclose(result["executor/action_args"], 1.0)

    def test_score_executor_type_with_pixel_square(self):
        """Test score method with executor type ground truth using pixel square metric."""
        ground_truth = {"thinking": "Test thinking", "action": "click(100, 200)"}

        prediction = (
            "<thinking>Test thinking</thinking><action>click(102, 198)</action>"
        )

        result = self.scorer.score(
            prediction, ground_truth, 
            coordinates_metric="pixel_square", coordinates_pixel_square_size=10
        )
        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["executor/action_type"] == 1
        assert np.isclose(result["executor/coordinates"], 1.0)

    def test_score_invalid_ground_truth(self):
        """Test score method with invalid ground truth."""
        ground_truth = {"invalid_key": "invalid_value"}

        prediction = "Some prediction"

        with pytest.raises(ValueError, match="Invalid ground truth type"):
            self.scorer.score(prediction, ground_truth)

    def test_compute_score_function(self):
        """Test the standalone compute_score function."""
        prediction = "<reasoning>Test</reasoning><should_end>true</should_end><goal_achieved>true</goal_achieved><answer>Test</answer>"
        ground_truth = {
            "reasoning": "Test",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "Test",
        }

        result = compute_score(prediction, ground_truth)

        assert "score" in result
        assert 0 <= result["score"] <= 1

    def test_compute_score_function_with_pixel_square(self):
        """Test the standalone compute_score function with pixel square metric."""
        prediction = "<thinking>Test</thinking><action>click(100, 200)</action>"
        ground_truth = {
            "thinking": "Test",
            "action": "click(105, 195)",
        }

        result = compute_score(
            prediction, ground_truth, 
            coordinates_metric="pixel_square", coordinates_pixel_square_size=20
        )

        assert "score" in result
        assert result["score"] == 1.0  # Should be perfect since coordinates are within 20x20 square

    def test_score_reward_model_with_none_values(self):
        """Test reward model scoring when extract returns None values."""
        prediction = "No valid tags"
        ground_truth = {
            "reasoning": "Expected reasoning",
            "should_end": "true",
            "goal_achieved": "true",
            "answer": "Expected answer",
        }

        result = self.scorer._score_reward_model(prediction, ground_truth, detailed=True)

        # All scores should be 0
        assert np.isclose(result["format"], 0.0)
        assert result["reward_model/should_end"] == 0
        assert result["reward_model/goal_achieved"] == 0
        assert result["reward_model/answer"] == 0

    def test_score_executor_with_bbox(self):
        """Test executor scoring with bbox in ground truth."""
        prediction = "<thinking>I need to click the button</thinking><action>click(100, 200)</action>"
        ground_truth = {
            "thinking": "I need to click the button",
            "action": "click(100, 200)",
            "bbox": [(90, 190, 110, 210)]  # bbox around the click point
        }

        result = self.scorer._score_executor(
            prediction, ground_truth, detailed=True, 
            coordinates_metric="gaussian", coordinates_gaussian_sigma=2, coordinates_pixel_square_size=5
        )

        assert np.isclose(result["score"], 1.0)
        assert np.isclose(result["format"], 1.0)
        assert result["executor/action_type"] == 1
        assert np.isclose(result["executor/coordinates"], 1.0)

    def test_coordinate_scoring_bbox_not_implemented(self):
        """Test that bbox metric raises NotImplementedError."""
        pred_coords = [(10, 20)]
        gt_coords = [(11, 21)]
        
        with pytest.raises(NotImplementedError, match="Bounding box distance metric not implemented"):
            self.scorer._calculate_coordinates_score(
                pred_coords, gt_coords, metric="bbox", 
                gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
            )

    def test_coordinate_scoring_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        pred_coords = [(10, 20)]
        gt_coords = [(11, 21)]
        
        with pytest.raises(ValueError, match="Invalid coordinate scoring metric: invalid"):
            self.scorer._calculate_coordinates_score(
                pred_coords, gt_coords, metric="invalid", 
                gaussian_sigma=2, pixel_square_size=5, gt_bbox=None
            )

    def test_gaussian_distance_score(self):
        """Test the Gaussian distance scoring function."""
        # Test identical coordinates
        score = self.scorer._calculate_gaussian_distance_score((10, 20), (10, 20), sigma=2)
        assert np.isclose(score, 1.0)
        
        # Test close coordinates
        score = self.scorer._calculate_gaussian_distance_score((10, 20), (11, 21), sigma=2)
        assert 0.7 < score < 1.0
        
        # Test far coordinates
        score = self.scorer._calculate_gaussian_distance_score((10, 20), (100, 200), sigma=2)
        assert score < 0.01
