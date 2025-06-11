"""
Reward scoring for UI Subtask task
"""

from difflib import SequenceMatcher
from typing import Literal

import numpy as np

from orby.utils.action_utils import get_action_info, extract_content_by_tags


class UISubtaskRewardScorer:
    """Reward scorer for UI Subtask task."""

    def __init__(self):
        super().__init__()
        self.reward_model_weights = {
            "format": 0.1,
            "should_end": 0.3,
            "goal_achieved": 0.3,
            "answer": 0.3,
        }
        self.executor_weights = {
            "format": 0.1,
            "action_type": 0.2,
            "coordinates": 0.5,
            "action_args": 0.2,
        }
        self.reward_model_tags = ["reasoning", "should_end", "goal_achieved", "answer"]
        self.executor_tags = ["thinking", "action"]

    def _check_text_similarity(
        self, pred_content: str, gt_content: str, threshold: float = 0.8
    ) -> bool:
        """Check if predicted content matches ground truth content using text similarity.

        Args:
            pred_content: Predicted content string
            gt_content: Ground truth content string
            threshold: Minimum similarity score to consider as match

        Returns:
            True if similarity score is above threshold
        """
        similarity = SequenceMatcher(
            None, pred_content.lower(), gt_content.lower()
        ).ratio()
        return similarity >= threshold

    def _score_reward_model(
        self, prediction: str, ground_truth: dict, detailed: bool
    ) -> dict:
        """
        Score the prediction against ground truth for reward model.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information.
                - reasoning: str (unused)
                - should_end: Literal["true", "false"]
                - goal_achieved: Literal["true", "false"]
                - answer: str

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - other individual check scores
        """
        pred_dict = extract_content_by_tags(prediction, self.reward_model_tags)

        # check format
        format_score = sum(
            1 for value in pred_dict.values() if value is not None
        ) / len(self.reward_model_tags)

        # Convert everything to string, even None
        pred_dict = {key: str(value) for key, value in pred_dict.items()}
        gt_dict = {key: str(value) for key, value in ground_truth.items()}

        pred_dict["should_end"] = pred_dict["should_end"].lower() == "true"
        pred_dict["goal_achieved"] = pred_dict["goal_achieved"].lower() == "true"
        gt_dict["should_end"] = gt_dict["should_end"].lower() == "true"
        gt_dict["goal_achieved"] = gt_dict["goal_achieved"].lower() == "true"

        # Remove gt answer if should_end is false
        if not gt_dict["should_end"]:
            gt_dict["answer"] = ""

        try:
            should_end_score = int(pred_dict["should_end"] == gt_dict["should_end"])
        except Exception as e:
            print(f"Error calculating should end score: {e}")
            should_end_score = 0
        try:
            goal_achieved_score = int(
                pred_dict["goal_achieved"] == gt_dict["goal_achieved"]
            )
        except Exception as e:
            print(f"Error calculating goal achieved score: {e}")
            goal_achieved_score = 0
        try:
            answer_score = int(
                self._check_text_similarity(pred_dict["answer"], gt_dict["answer"])
            )
        except Exception as e:
            print(f"Error calculating answer score: {e}")
            answer_score = 0

        score = (
            format_score * self.reward_model_weights["format"]
            + should_end_score * self.reward_model_weights["should_end"]
            + goal_achieved_score * self.reward_model_weights["goal_achieved"]
            + answer_score * self.reward_model_weights["answer"]
        )

        if detailed:
            details = {
                "score": score,
                "format": format_score,
                "reward_model/reward": score,
                "reward_model/format": format_score,
                "reward_model/should_end": should_end_score,
                "reward_model/goal_achieved": goal_achieved_score,
                "reward_model/answer": answer_score,
            }
        else:
            details = {
                "score": score,
                "format": format_score,
                "executor/action_type": 0,
                "executor/coordinates": 0,
                "executor/action_args": 0,
                "reward_model/should_end": should_end_score,
                "reward_model/goal_achieved": goal_achieved_score,
                "reward_model/answer": answer_score,
            }

        return details

    def _calculate_coordinates_score(
        self,
        pred_coordinates: list[tuple[float, float]] | None,
        gt_coordinates: list[tuple[float, float]] | None,
        metric: Literal["gaussian", "pixel_square", "bbox"],
        gaussian_sigma: float,
        pixel_square_size: int,
        gt_bbox: list[tuple[float, float, float, float]] | None,
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
        # If coordinates are not necessary and the model predicted None
        # we should not penalize the model
        if pred_coordinates is None and gt_coordinates is None:
            return 1.0
        # If the model predicted None but the ground truth is not None, or vice versa,
        # we should penalize the model
        if pred_coordinates is None or gt_coordinates is None:
            return 0.0

        # If the length of the coordinates is different, we should penalize the model
        if len(pred_coordinates) != len(gt_coordinates):
            return 0.0

        scores = []
        for pred_coord, gt_coord in zip(pred_coordinates, gt_coordinates):
            if metric == "gaussian":
                score = self._calculate_gaussian_distance_score(
                    pred_coord, gt_coord, gaussian_sigma
                )
            elif metric == "pixel_square":
                score = self._calculate_pixel_square_score(
                    pred_coord, gt_coord, pixel_square_size
                )
            elif metric == "bbox":
                # score = self._calculate_bbox_score(pred_coord, gt_bbox)
                raise NotImplementedError(
                    "Bounding box distance metric not implemented"
                )
            else:
                raise ValueError(f"Invalid coordinate scoring metric: {metric}")
            scores.append(score)

        return np.mean(scores)

    def _calculate_gaussian_distance_score(
        self,
        pred_coord: tuple[float, float],
        gt_coord: tuple[float, float],
        sigma: float,
    ) -> float:
        """
        Calculate the score for the coordinates using a Gaussian similarity score.
        """
        pred = np.asarray(pred_coord)
        truth = np.asarray(gt_coord)
        d2 = np.sum((pred - truth) ** 2)
        score = np.exp(-d2 / (2 * sigma**2))
        return score

    def _calculate_pixel_square_score(
        self,
        pred_coord: tuple[float, float],
        gt_coord: tuple[float, float],
        pixel_square_size: int,
    ) -> float:
        """
        Calculate the score for the coordinates using a pixel square distance score.
        """
        gt_x1, gt_y1, gt_x2, gt_y2 = [
            max(gt_coord[0] - pixel_square_size / 2, 0),
            max(gt_coord[1] - pixel_square_size / 2, 0),
            gt_coord[0] + pixel_square_size / 2,
            gt_coord[1] + pixel_square_size / 2,
        ]

        in_bbox = (
            pred_coord[0] >= gt_x1
            and pred_coord[0] <= gt_x2
            and pred_coord[1] >= gt_y1
            and pred_coord[1] <= gt_y2
        )

        return float(in_bbox)

    def _calculate_action_args_score(
        self, pred_args: dict[str, str] | None, gt_args: dict[str, str] | None
    ) -> float:
        """
        Calculate the score for the action arguments.
        We treat every argument as a string and use the textual similarity score.

        Args:
            pred_args: Predicted arguments
            gt_args: Ground truth arguments

        Returns:
            Score for the arguments between 0 and 1
        """
        # If both are None, we should not penalize the model
        if pred_args is None and gt_args is None:
            return 1.0
        # If one is None but the other is not, we should penalize the model
        if pred_args is None or gt_args is None:
            return 0.0
        # if there is a key mismatch, we should penalize the model
        if pred_args.keys() != gt_args.keys():
            return 0.0

        scores = []
        for key in pred_args.keys():
            score = self._check_text_similarity(pred_args[key], gt_args[key])
            scores.append(score)

        return np.mean(scores)

    def _score_executor(
        self,
        prediction: str,
        ground_truth: dict,
        detailed: bool,
        coordinates_metric: Literal["gaussian", "pixel_square", "bbox"],
        coordinates_gaussian_sigma: float,
        coordinates_pixel_square_size: int,
    ) -> dict:
        """
        Score the prediction against ground truth for executor.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information.
                - thinking: str (unused)
                - action: str

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - other individual check scores
        """
        pred_dict = extract_content_by_tags(prediction, self.executor_tags)
        format_score = sum(
            1 for value in pred_dict.values() if value is not None
        ) / len(self.executor_tags)

        gt_bbox = ground_truth.get("bbox", None)
        assert gt_bbox is None or (
            isinstance(gt_bbox, list)
            and all(
                isinstance(item, (tuple, list)) and len(item) == 4 for item in gt_bbox
            )
        ), "Ground truth bbox must be a list of tuples of 4 floats or None"

        # Convert everything to string, even None
        pred_dict = {key: str(value) for key, value in pred_dict.items()}
        gt_dict = {key: str(value) for key, value in ground_truth.items()}

        action_type_parser_error = False
        try:
            pred_action_info = get_action_info(pred_dict["action"])
            gt_action_info = get_action_info(gt_dict["action"])
            action_type_score = int(
                pred_action_info["action_type"] == gt_action_info["action_type"]
            )
        except Exception as e:
            # If the action is not in the action space, we should penalize the model and exit early
            print(f"Error with action type parsing: {e}")
            action_type_parser_error = True
            action_type_score = 0

        coordinates_score = 0
        action_args_score = 0
        if not action_type_parser_error:
            try:
                coordinates_score = self._calculate_coordinates_score(
                    pred_action_info["coordinates"],
                    gt_action_info["coordinates"],
                    metric=coordinates_metric,
                    gaussian_sigma=coordinates_gaussian_sigma,
                    pixel_square_size=coordinates_pixel_square_size,
                    gt_bbox=gt_bbox,
                )
            except Exception as e:
                print(f"Error calculating coordinates score: {e}")
            try:
                action_args_score = self._calculate_action_args_score(
                    pred_action_info["args"], gt_action_info["args"]
                )
            except Exception as e:
                print(f"Error calculating action args score: {e}")

        score = (
            format_score * self.executor_weights["format"]
            + action_type_score * self.executor_weights["action_type"]
            + coordinates_score * self.executor_weights["coordinates"]
            + action_args_score * self.executor_weights["action_args"]
        )

        if detailed:
            details = {
                "score": score,
                "format": format_score,
                "executor/score": score,
                "executor/format": format_score,
                "executor/action_type": action_type_score,
                "executor/coordinates": coordinates_score,
                "executor/action_args": action_args_score,
                "executor/in_action_space": not action_type_parser_error,
            }

            # Aggregate action-type-wise scores
            if not action_type_parser_error:
                details[f"executor/coordinates/{gt_action_info['action_type']}"] = (
                    coordinates_score
                )
                details[f"executor/action_args/{gt_action_info['action_type']}"] = (
                    action_args_score
                )
        else:
            details = {
                "score": score,
                "format": format_score,
                "executor/action_type": action_type_score,
                "executor/coordinates": coordinates_score,
                "executor/action_args": action_args_score,
                "reward_model/should_end": 0,
                "reward_model/goal_achieved": 0,
                "reward_model/answer": 0,
            }

        return details

    def score(
        self,
        prediction: str,
        ground_truth: dict,
        detailed: bool,
        coordinates_metric: Literal["gaussian", "pixel_square", "bbox"],
        coordinates_gaussian_sigma: float,
        coordinates_pixel_square_size: int,
    ) -> dict:
        """Score the prediction against ground truth.

        Args:
            prediction: Model prediction string
            ground_truth: Dictionary containing ground truth information.
            There can be 2 types of ground truth:
            - reward_model: with fields
                - reasoning: str (unused)
                - should_end: Literal["true", "false"]
                - goal_achieved: Literal["true", "false"]
                - answer: str
            - executor: with fields
                - thinking: str (unused)
                - action: str
                - bbox: list[tuple[float, float, float, float]] | None
            detailed (bool): Whether to return detailed, categorized scores; this is mostly for
                offline evaluation after training
            coordinates_metric (Literal["gaussian", "pixel_square", "bbox"]): Metric to use for coordinates scoring
                - gaussian: Gaussian distance metric
                - pixel_square: Pixel square distance metric
                - bbox: Bounding box distance metric (requires additional bbox information in ground truth)
            coordinates_gaussian_sigma (float): Sigma for Gaussian distance metric
            coordinates_pixel_square_size (int): width and height of the square bbox around the
                ground truth pixel for pixel square distance metric

        Returns:
            Dictionary containing:
                - score: Overall score (0-1)
                - other individual check scores
        """
        gt_keys = ground_truth.keys()
        if (
            "reasoning" in gt_keys
            and "should_end" in gt_keys
            and "goal_achieved" in gt_keys
            and "answer" in gt_keys
            and ("action" not in gt_keys or not ground_truth["action"])
            and ("thinking" not in gt_keys or not ground_truth["thinking"])
        ):
            result = self._score_reward_model(prediction, ground_truth, detailed)
        elif (
            "thinking" in gt_keys
            and "action" in gt_keys
            and ("reasoning" not in gt_keys or not ground_truth["reasoning"])
            and ("should_end" not in gt_keys or not ground_truth["should_end"])
            and ("goal_achieved" not in gt_keys or not ground_truth["goal_achieved"])
            and ("answer" not in gt_keys or not ground_truth["answer"])
        ):
            result = self._score_executor(
                prediction,
                ground_truth,
                detailed,
                coordinates_metric=coordinates_metric,
                coordinates_gaussian_sigma=coordinates_gaussian_sigma,
                coordinates_pixel_square_size=coordinates_pixel_square_size,
            )
        else:
            raise ValueError("Invalid ground truth type")

        return result


def compute_score(
    prediction: str,
    ground_truth: dict,
    detailed: bool,
    coordinates_metric: Literal["gaussian", "pixel_square", "bbox"],
    coordinates_gaussian_sigma: float,
    coordinates_pixel_square_size: int,
) -> dict:
    """Compute score for a single prediction.

    Args:
        prediction (str): Prediction string
        ground_truth (dict): Dictionary containing ground truth information
        detailed (bool): Whether to return detailed, categorized scores; this is mostly for
            offline evaluation after training
        coordinates_metric (Literal["gaussian", "pixel_square", "bbox"]): Metric to use for coordinates scoring
            - gaussian: Gaussian distance metric
            - pixel_square: Pixel square distance metric
            - bbox: Bounding box distance metric (requires additional bbox information in ground truth)
        coordinates_gaussian_sigma (float): Sigma for Gaussian distance metric
        coordinates_pixel_square_size (int): width and height of the square bbox around the
            ground truth pixel for pixel square distance metric
    """
    scorer = UISubtaskRewardScorer()
    result = scorer.score(
        prediction,
        ground_truth,
        detailed=detailed,
        coordinates_metric=coordinates_metric,
        coordinates_gaussian_sigma=coordinates_gaussian_sigma,
        coordinates_pixel_square_size=coordinates_pixel_square_size,
    )
    return result


def training_reward_func(
    data_source,
    solution_str,
    ground_truth,
    coordinates_metric="gaussian",
    coordinates_gaussian_sigma=5,
    coordinates_pixel_square_size=10,
    extra_info=None,
):
    if data_source == "subtask_direct_distill":
        from orby.reward import subtask

        return subtask.compute_score(
            solution_str,
            ground_truth,
            detailed=False,
            coordinates_metric=coordinates_metric,
            coordinates_gaussian_sigma=coordinates_gaussian_sigma,
            coordinates_pixel_square_size=coordinates_pixel_square_size,
        )
    else:
        raise NotImplementedError


def eval_reward_func(
    data_source,
    solution_str,
    ground_truth,
    coordinates_metric="gaussian",
    coordinates_gaussian_sigma=5,
    coordinates_pixel_square_size=10,
    extra_info=None,
):
    if data_source == "subtask_direct_distill":
        from orby.reward import subtask

        return subtask.compute_score(
            solution_str,
            ground_truth,
            detailed=True,
            coordinates_metric=coordinates_metric,
            coordinates_gaussian_sigma=coordinates_gaussian_sigma,
            coordinates_pixel_square_size=coordinates_pixel_square_size,
        )
    else:
        raise NotImplementedError
