from types import SimpleNamespace
import numpy as np
import torch
from verl import DataProto
from verl.trainer.config import ProbeCreditConfig
from verl.workers.config.rollout import BoundaryReturnConfig
from verl.experimental.natural_continuation_boundary_return.dapo_trainer import RayDAPOBoundaryReturnTrainer
ForcedAnswerProbeConfig = SimpleNamespace

def _config(mode="replace"):
    boundary = BoundaryReturnConfig(
        mode=mode,
        long_response_length=8,
        max_concurrent_requests=2,
        request_batch_size=2,
    )
    algorithm = SimpleNamespace(
        adv_estimator="grpo",
        use_kl_in_reward=False,
        rollout_correction=None,
        filter_groups=SimpleNamespace(enable=True, metric="acc", max_num_gen_batches=3),
        probe_credit=ProbeCreditConfig(enable=False, coef=0.0),
        censor_aware_advantage=SimpleNamespace(enable=False),
        readiness_dominance=SimpleNamespace(mode="off"),
        success_support_floor=SimpleNamespace(mode="off"),
        on_policy_budgeted_capability_floor=SimpleNamespace(mode="off"),
    )
    algorithm.get = lambda name, default=None: getattr(algorithm, name, default)
    rollout = SimpleNamespace(
        name="vllm",
        mode="async",
        response_length=4,
        prompt_length=2,
        max_model_len=10,
        ignore_eos=False,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        repetition_penalty=1.0,
        calculate_log_probs=False,
        n=2,
        multi_turn=SimpleNamespace(enable=False),
        agent=SimpleNamespace(
            default_agent_loop="single_turn_agent",
            agent_loop_config_path=None,
            agent_loop_manager_class=None,
            custom_async_server=SimpleNamespace(path=None, name=None),
        ),
        forced_answer_probe=ForcedAnswerProbeConfig(enable=False),
        boundary_return=boundary,
    )
    config = SimpleNamespace(
        algorithm=algorithm,
        actor_rollout_ref=SimpleNamespace(rollout=rollout),
        critic=SimpleNamespace(enable=False),
        reward=SimpleNamespace(
            reward_manager=SimpleNamespace(source="register", name="dapo"),
            custom_reward_function=SimpleNamespace(path=None),
            reward_model=SimpleNamespace(enable=False),
            sandbox_fusion=SimpleNamespace(url=None),
        ),
        distillation=SimpleNamespace(enabled=False),
        global_profiler=SimpleNamespace(steps=None),
    )
    return config

def _trainer(mode="replace"):
    trainer = object.__new__(RayDAPOBoundaryReturnTrainer)
    trainer.config = _config(mode)
    trainer.use_critic = False
    trainer.use_teacher_policy = False
    return trainer

def _hook_candidate() -> DataProto:
    prompts = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
    responses = torch.tensor([[3, 4, 5, 6], [7, 8, 9, 10]], dtype=torch.long)
    mask = torch.ones((2, 6), dtype=torch.long)
    scores = torch.zeros((2, 4), dtype=torch.float32)
    return DataProto.from_dict(
        tensors={
            "prompts": prompts,
            "responses": responses,
            "input_ids": torch.cat((prompts, responses), -1),
            "attention_mask": mask,
            "position_ids": torch.arange(6).repeat(2, 1),
            "response_mask": torch.ones((2, 4), dtype=torch.long),
            "token_level_scores": scores.clone(),
            "token_level_rewards": scores.clone(),
        },
        non_tensors={
            "uid": np.asarray(["u", "u"], dtype=object),
            "trajectory_id": np.asarray(["u:0", "u:1"], dtype=object),
            "rollout_policy_version": np.asarray([7, 7], dtype=object),
            "finish_reason": np.asarray(["length", "length"], dtype=object),
            "acc": np.asarray([0.0, 1.0]),
            "score": np.asarray([0.0, 1.0]),
            "data_source": np.asarray(["math", "math"], dtype=object),
            "reward_model": np.asarray([{"ground_truth": "0"}] * 2, dtype=object),
        },
        meta_info={"reward_extra_keys": ["acc", "score"]},
    )
