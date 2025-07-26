"""Reward function that supports multiple tasks."""


def training_reward_func(
    data_source,
    solution_str,
    ground_truth,
    **kwargs,
):
    if data_source in ["uground"]:
        from orby.reward import uground

        return uground.reward_func(
            data_source,
            solution_str,
            ground_truth,
            kwargs.get("prompt_format"),
            kwargs.get("use_gaussian"),
            kwargs.get("extra_info"),
        )
    elif data_source in ["subtask_direct_distill"]:
        from orby.reward import subtask

        return subtask.training_reward_func(
            data_source,
            solution_str,
            ground_truth,
            kwargs.get("coordinates_metric", "gaussian"),
            kwargs.get("coordinates_gaussian_sigma", 5),
            kwargs.get("coordinates_pixel_square_size", 10),
            kwargs.get("extra_info"),
        )
    else:
        raise NotImplementedError


def eval_reward_func(
    data_source,
    solution_str,
    ground_truth,
    **kwargs,
):
    if data_source in ["uground"]:
        from orby.reward import uground

        return uground.reward_func(
            data_source,
            solution_str,
            ground_truth,
            kwargs.get("prompt_format"),
            kwargs.get("use_gaussian"),
            kwargs.get("extra_info"),
        )
    elif data_source in ["subtask_direct_distill"]:
        from orby.reward import subtask

        return subtask.eval_reward_func(
            data_source,
            solution_str,
            ground_truth,
            kwargs.get("coordinates_metric", "gaussian"),
            kwargs.get("coordinates_gaussian_sigma", 5),
            kwargs.get("coordinates_pixel_square_size", 10),
            kwargs.get("extra_info"),
        )
    else:
        raise NotImplementedError
