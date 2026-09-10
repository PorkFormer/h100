"""Shared NCBR preflight, with DAPO-only filter constraints explicit."""
from typing import Any
from verl.trainer.ppo.core_algos import AdvantageEstimator
from .config import resolve_boundary_config, config_get as _config_get

def validate_boundary_return_preflight(config: Any, *, use_critic: bool | None = None, require_dynamic_filter: bool = True) -> None:
    """Pure, fail-closed validation safe to run before any data or worker setup."""
    rollout = _config_get(_config_get(config, "actor_rollout_ref"), "rollout")
    boundary = resolve_boundary_config(config)
    boundary.validate()
    gate_cycles = int(_config_get(_config_get(config, "trainer"), "dynamic_sampling_gate_cycles", 0))
    if gate_cycles not in (0, 3):
        raise ValueError("Dynamic Sampling Gate 0 requires exactly 3 cycles or must be disabled with 0")
    if boundary.mode == "off":
        return

    algorithm = _config_get(config, "algorithm")
    if _config_get(algorithm, "adv_estimator") not in (
        "grpo",
        AdvantageEstimator.GRPO,
        AdvantageEstimator.GRPO_VECTORIZED,
    ):
        raise ValueError("boundary_return supports synchronous GRPO only")
    if _config_get(rollout, "name") != "vllm":
        raise ValueError("boundary_return supports the vLLM rollout backend only")
    if _config_get(rollout, "mode") != "async":
        raise ValueError("boundary_return requires rollout.mode=async")
    if use_critic is None:
        use_critic = bool(_config_get(_config_get(config, "critic"), "enable", False))
    if use_critic:
        raise ValueError("boundary_return requires GRPO with no critic")
    if bool(_config_get(rollout, "ignore_eos", False)):
        raise ValueError("boundary_return requires ignore_eos=false")
    if bool(_config_get(_config_get(rollout, "multi_turn"), "enable", False)):
        raise ValueError("boundary_return supports single-turn rollout only")
    if bool(_config_get(algorithm, "use_kl_in_reward", False)):
        raise ValueError("boundary_return requires algorithm.use_kl_in_reward=false")

    agent = _config_get(rollout, "agent")
    if _config_get(agent, "default_agent_loop") != "single_turn_agent":
        raise ValueError("boundary_return requires agent.default_agent_loop=single_turn_agent")
    if (
        _config_get(agent, "agent_loop_config_path") is not None
        or _config_get(agent, "agent_loop_manager_class") is not None
    ):
        raise ValueError("boundary_return does not support a custom agent loop")
    custom_server = _config_get(agent, "custom_async_server")
    if _config_get(custom_server, "path") is not None or _config_get(custom_server, "name") is not None:
        raise ValueError("boundary_return does not support a custom async server")

    reward = _config_get(config, "reward")
    reward_manager = _config_get(reward, "reward_manager")
    if _config_get(reward_manager, "source") != "register":
        raise ValueError("boundary_return requires the registered DAPO reward manager")
    if _config_get(reward_manager, "name") != "dapo":
        raise ValueError("boundary_return requires the DAPO reward manager")
    if boundary.correctness_key != "acc":
        raise ValueError("boundary_return v1 requires correctness_key=acc")
    if boundary.task_score_key != "score":
        raise ValueError("boundary_return v1 requires task_score_key=score")
    if _config_get(_config_get(reward, "custom_reward_function"), "path") is not None:
        raise ValueError("boundary_return v1 does not support a custom reward function path")
    if bool(_config_get(_config_get(reward, "reward_model"), "enable", False)):
        raise ValueError("boundary_return v1 does not support the reward model path")
    if _config_get(_config_get(reward, "sandbox_fusion"), "url") is not None:
        raise ValueError("boundary_return v1 does not support the sandbox reward path")
    if bool(_config_get(_config_get(config, "distillation"), "enabled", False)):
        raise ValueError("boundary_return v1 does not support distillation or a teacher policy")
    rollout_correction = _config_get(algorithm, "rollout_correction")
    if rollout_correction is not None and any(
        (
            _config_get(rollout_correction, "rollout_is") is not None,
            _config_get(rollout_correction, "rollout_rs") is not None,
            bool(_config_get(rollout_correction, "bypass_mode", False)),
        )
    ):
        raise ValueError("boundary_return v1 does not support rollout correction")
    if _config_get(_config_get(config, "global_profiler"), "steps"):
        raise ValueError("boundary_return v1 does not support configured profiling steps")

    short_length = int(_config_get(rollout, "response_length"))
    if boundary.long_response_length <= short_length:
        raise ValueError("boundary_return requires L > H (long_response_length > response_length)")
    max_model_len = _config_get(rollout, "max_model_len")
    prompt_length = int(_config_get(rollout, "prompt_length"))
    if max_model_len is None or prompt_length + boundary.long_response_length > int(max_model_len):
        raise ValueError("boundary_return context requires max_model_len >= prompt_length + long_response_length")

    filter_groups = _config_get(algorithm, "filter_groups")
    if require_dynamic_filter and boundary.mode == "replace":
        if not bool(_config_get(filter_groups, "enable", False)):
            raise ValueError("boundary_return replace requires filter_groups.enable=true")
        if _config_get(filter_groups, "metric") != boundary.correctness_key:
            raise ValueError("boundary_return replace requires Hydra filter_groups.metric == correctness_key")

    forced_answer = _config_get(rollout, "forced_answer_probe")
    forced_credit = _config_get(forced_answer, "training_credit")
    if bool(_config_get(forced_answer, "enable", False)) or bool(_config_get(forced_credit, "enable", False)):
        raise ValueError("boundary_return cannot be combined with forced-answer or FA-TR")
    if bool(_config_get(_config_get(algorithm, "censor_aware_advantage"), "enable", False)):
        raise ValueError("boundary_return cannot be combined with FA-CAC/FA-RAR")
    if bool(_config_get(_config_get(algorithm, "probe_credit"), "enable", False)):
        raise ValueError("boundary_return cannot be combined with Probe Credit")
    if _config_get(_config_get(algorithm, "readiness_dominance"), "mode", "off") != "off":
        raise ValueError("boundary_return cannot be combined with Readiness")
    if _config_get(_config_get(algorithm, "success_support_floor"), "mode", "off") != "off":
        raise ValueError("boundary_return cannot be combined with BSSF")
    if _config_get(_config_get(algorithm, "on_policy_budgeted_capability_floor"), "mode", "off") != "off":
        raise ValueError("boundary_return cannot be combined with OBCF")
