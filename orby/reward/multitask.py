"""Reward function that supports multiple tasks."""


def training_reward_func(
    data_source,
    solution_str,
    ground_truth,
    **kwargs,
):
    if data_source in ["uground"]:
        from orby.reward import uground

        return uground.reward_func(data_source, solution_str, ground_truth, **kwargs)
    elif data_source in ["subtask"]:
        from orby.reward import subtask

        return subtask.training_reward_func(
            data_source, solution_str, ground_truth, **kwargs
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

        return uground.reward_func(data_source, solution_str, ground_truth, **kwargs)
    elif data_source in ["subtask"]:
        from orby.reward import subtask

        return subtask.eval_reward_func(
            data_source, solution_str, ground_truth, **kwargs
        )
    else:
        raise NotImplementedError
