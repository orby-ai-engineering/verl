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
Reward scoring for UI UGround task
"""

import re
import json
from typing import Dict, List, Tuple, Optional
import numpy as np


class UGroundRewardScorer:
    """Reward scorer for UI UGround task."""

    def __init__(self):
        super().__init__()
        self.thinking_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
        self.answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
        self.coordinate_pattern = re.compile(r"click\((\d*\.?\d+)[,\s]+(\d*\.?\d+)\)")

    def _extract_coordinates(
        self, answer_str: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract x, y coordinates from answer string.

        Args:
            answer_str: Answer string in format "x y"

        Returns:
            Tuple of (x, y) coordinates or (None, None) if not found
        """
        match = self.coordinate_pattern.match(answer_str.strip())
        if not match:
            return None, None
        x, y = match.groups()
        return float(x), float(y)

    def _check_coordinates_in_bbox(
        self, x: float, y: float, bbox: List[float], tolerance: float = 0.0
    ) -> bool:
        """Check if coordinates are within bounding box with tolerance.

        Args:
            x: X coordinate
            y: Y coordinate
            bbox: Bounding box [x1, y1, x2, y2]
            tolerance: Coordinate tolerance for matching

        Returns:
            True if coordinates are within bbox with tolerance
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 - tolerance <= x <= x2 + tolerance
            and y1 - tolerance <= y <= y2 + tolerance
        )

    def score(self, prediction: str, ground_truth: Dict) -> Dict:
        """Score the prediction against ground truth.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information
                - bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - details: Dictionary with individual check results
        """
        # Check 1: Format validation
        has_thinking = bool(self.thinking_pattern.search(prediction))
        answer_match = self.answer_pattern.search(prediction)
        has_answer = bool(answer_match)

        # Check 2: Coordinate validation
        pred_answer = answer_match.group(1).strip() if answer_match else ""
        pred_x, pred_y = self._extract_coordinates(pred_answer)

        bbox = ground_truth.get("bbox")

        coordinates_correct = False
        if bbox is not None and pred_x is not None and pred_y is not None:
            coordinates_correct = self._check_coordinates_in_bbox(pred_x, pred_y, bbox)

        # Calculate overall score
        format_score = 1.0 if (has_thinking and has_answer) else 0.0
        coord_score = 1.0 if coordinates_correct else 0.0

        # Weight the scores (can be adjusted based on importance)
        weights = {"format": 0.0, "coordinates": 1.0}

        overall_score = (
            weights["format"] * format_score + weights["coordinates"] * coord_score
        )

        details = {
            "score": overall_score,
            "has_thinking": has_thinking,
            "has_answer": has_answer,
            "format_score": format_score,
            "coordinates_predicted": (
                str((pred_x, pred_y))
                if pred_x is not None and pred_y is not None
                else "None"
            ),
            "coordinates_ground_truth": str(bbox),
            "coordinates_score": coord_score,
        }

        return details


class QwenUGroundScorer:
    """Reward scorer for UI UGround task with Qwen tool call format."""

    def __init__(self):
        super().__init__()
        # Qwen default response is like this:
        # <tool_call>
        # {"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [1240, 783]}}
        # </tool_call>
        # We need to extract the coordinate from the tool call.
        self.tool_call_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

    def _extract_coordinates(
        self, response: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Extract x, y coordinates from tool call response.

        Args:
            response: Response string containing tool call JSON

        Returns:
            Tuple of (x, y) coordinates or (None, None) if not found
        """
        match = self.tool_call_pattern.search(response)
        if not match:
            return None, None

        try:
            tool_call = json.loads(match.group(1).strip())
            if (
                tool_call.get("name") == "computer_use"
                and "arguments" in tool_call
                and "coordinate" in tool_call["arguments"]
            ):
                x, y = tool_call["arguments"]["coordinate"]
                return float(x), float(y)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        return None, None

    def _check_coordinates_in_bbox(
        self, x: float, y: float, bbox: List[float], tolerance: float = 0.0
    ) -> bool:
        """Check if coordinates are within bounding box with tolerance.

        Args:
            x: X coordinate
            y: Y coordinate
            bbox: Bounding box [x1, y1, x2, y2]
            tolerance: Coordinate tolerance for matching

        Returns:
            True if coordinates are within bbox with tolerance
        """
        x1, y1, x2, y2 = bbox
        return (
            x1 - tolerance <= x <= x2 + tolerance
            and y1 - tolerance <= y <= y2 + tolerance
        )

    def _calculate_gaussian_score(
        self,
        pred_coordinates: list[tuple[float, float]] | None,
        gt_coordinates: list[tuple[float, float]] | None,
    ) -> float:
        """
        Calculate the score for the coordinates of the action.
        We use a Gaussian similarity score with a sigma of 2 for all coordinates.

        Args:
            pred_coordinates: Predicted coordinates
            gt_coordinates: Ground truth coordinates

        Returns:
            Score for the coordinates between 0 and 1
        """
        # If the length of the coordinates is different, we should penalize the model
        if len(pred_coordinates) != len(gt_coordinates):
            return 0.0

        scores = []
        for pred_coord, gt_coord in zip(pred_coordinates, gt_coordinates):
            # Use a Gaussian similarity score with a sigma of 2
            sigma = 5
            pred = np.asarray(pred_coord)
            truth = np.asarray(gt_coord)
            d2 = np.sum((pred - truth) ** 2)
            score = np.exp(-d2 / (2 * sigma**2))
            scores.append(score)

        return np.mean(scores)

    def score(
        self, prediction: str, ground_truth: Dict, use_gaussian: bool = False
    ) -> Dict:
        """Score the prediction against ground truth.

        Args:
            prediction: Model prediction string with tool call
            ground_truth: Dictionary containing ground truth information
                - bbox: Bounding box [x1, y1, x2, y2]
            use_gaussian: Whether to use Gaussian scoring instead of bbox check

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - details: Dictionary with individual check results
        """
        # Check 1: Format validation
        has_tool_call = bool(self.tool_call_pattern.search(prediction))

        # Check 2: Coordinate validation
        pred_x, pred_y = self._extract_coordinates(prediction)

        bbox = ground_truth.get("bbox")

        coordinates_correct = False
        coord_score = 0.0
        if bbox is not None and pred_x is not None and pred_y is not None:
            if use_gaussian:
                # Convert single coordinate to list format for Gaussian scoring
                pred_coords = [(pred_x, pred_y)]
                # Convert bbox to list of coordinates (center point)
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                gt_coords = [(center_x, center_y)]
                coord_score = self._calculate_gaussian_score(pred_coords, gt_coords)
            else:
                coordinates_correct = self._check_coordinates_in_bbox(
                    pred_x, pred_y, bbox
                )
                coord_score = 1.0 if coordinates_correct else 0.0

        # Calculate overall score
        format_score = 1.0 if has_tool_call else 0.0

        # Weight the scores (can be adjusted based on importance)
        weights = {"format": 0.2, "coordinates": 0.8}

        overall_score = (
            weights["format"] * format_score + weights["coordinates"] * coord_score
        )

        details = {
            "score": overall_score,
            "has_tool_call": has_tool_call,
            "format_score": format_score,
            "coordinates_predicted": (
                str((pred_x, pred_y))
                if pred_x is not None and pred_y is not None
                else "None"
            ),
            "coordinates_ground_truth": str(bbox),
            "coordinates_score": coord_score,
            "scoring_method": "gaussian" if use_gaussian else "bbox",
        }

        return details


def compute_score(prediction: str, ground_truth: Dict) -> Dict:
    """Compute score for a single prediction.

    Args:
        prediction: Prediction string
        ground_truth: Dictionary containing ground truth information
            - bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Dictionary containing:
            - score: Overall score (0-1)
            - details: Dictionary with individual check results
    """
    scorer = UGroundRewardScorer()
    result = scorer.score(prediction, ground_truth)
    return result


def compute_qwen_score(
    prediction: str, ground_truth: Dict, use_gaussian: bool = False
) -> Dict:
    """Compute score for a single prediction in Qwen format.

    Args:
        prediction: Prediction string with tool call
        ground_truth: Dictionary containing ground truth information
            - bbox: Bounding box [x1, y1, x2, y2]
        use_gaussian: Whether to use Gaussian scoring instead of bbox check

    Returns:
        Dictionary containing:
            - score: Overall score (0-1)
            - details: Dictionary with individual check results
    """
    scorer = QwenUGroundScorer()
    result = scorer.score(prediction, ground_truth, use_gaussian=use_gaussian)
    return result


def reward_func(
    data_source,
    solution_str,
    ground_truth,
    prompt_format=None,
    use_gaussian=False,
    extra_info=None,
):
    if data_source in ["uground"]:
        from orby.reward import uground

        # Check if we should use Qwen format scoring
        if prompt_format == "qwen":
            return uground.compute_qwen_score(
                solution_str, ground_truth, use_gaussian=use_gaussian
            )
        return uground.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
