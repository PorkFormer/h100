"""Synchronous PPPO with complete branch scoring and committed schedule state."""
import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from verl.experimental.agent_loop.agent_loop import build_rollout_sampling_params
from verl.experimental.probe_credit.dapo_trainer import RayDAPOProbeCreditTrainer
from verl.experimental.pppo.config import validate_config
from verl.experimental.pppo.objective import group_statistics
from verl.experimental.pppo.prefix import prefix_lengths, actor_batch
from verl.experimental.pppo.runtime import build_requests, generate_chunk
from verl.experimental.pppo.scoring import scoring_batch, correctness
from verl.experimental.pppo.schedule import PrefixSchedule


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


class RayPPPOTrainer(RayDAPOProbeCreditTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        validate_config(self.config)
        self.pppo = self.config.algorithm.pppo
        self.prefix_schedule = PrefixSchedule(eta=int(self.pppo.eta_percent))
        self.occurrences = 0
        self.data_epoch = 0
        self.epoch_batches = 0
        self.pending_checkpoint = None
        self.output = Path(self.config.trainer.default_local_dir).parent
        self.output.mkdir(parents=True, exist_ok=True)
        self._last_validation = -1
        # Operational restart/stop/output fields do not change the mathematical experiment.
        resolved = OmegaConf.to_container(self.config, resolve=True)
        for key in ("resume_mode", "resume_from_path", "default_local_dir"):
            resolved["trainer"].pop(key, None)
        resolved["algorithm"]["pppo"].pop("stop_after_step", None)
        self.config_hash = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
        atomic_json(self.output / "pppo_manifest.json", {"config_hash": self.config_hash,
                    "holdout_hash": self.pppo.holdout_hash, "module": __file__,
                    "normalization": "question_original_length", "total_response_cap": self.pppo.response_cap,
                    "std_correction": 1, "epsilon": 1e-6, "seed": self.pppo.seed})

    def _resume_data_epoch(self):
        return self.data_epoch

    def _initialize_candidate_identity(self, candidate, epoch):
        if epoch != self.data_epoch:
            self.data_epoch, self.epoch_batches = epoch, 0
        uids = []
        for i in range(len(candidate)):
            raw = candidate.non_tensor_batch["raw_prompt"][i]
            token_hash = hashlib.sha256(json.dumps(raw.tolist() if hasattr(raw, "tolist") else raw, sort_keys=True, default=lambda x: x.tolist()).encode()).hexdigest()
            uids.append(f"pppo:{self.pppo.seed}:{epoch}:{self.occurrences}:{token_hash}")
            self.occurrences += 1
        candidate.non_tensor_batch["uid"] = np.asarray(uids, dtype=object)
        self.epoch_batches += 1

    def _effective_filter_metric(self):
        return "pppo_filter_reward"

    def _process_candidate_after_reward_before_filter(self, candidate, metrics, timing_raw, generation_batch_index):
        started = time.perf_counter()
        version = self._validate_rollout_policy_version(candidate)
        lengths, h = prefix_lengths(candidate.batch["response_mask"], self.prefix_schedule.eta)
        candidate.batch["pppo_original_length"] = lengths
        candidate.batch["pppo_prefix_length"] = h
        original_acc = correctness(candidate.non_tensor_batch, len(candidate))
        rewards = original_acc.copy()
        counts = np.zeros(len(candidate), dtype=np.int64)
        requests = build_requests(candidate, int(self.pppo.extra_continuations_per_prefix), int(self.pppo.response_cap),
                                  version, int(self.pppo.seed), int(self.config.actor_rollout_ref.rollout.max_model_len))
        tail_tokens = input_tokens = 0
        seen = set()
        sampling = build_rollout_sampling_params(self.config.actor_rollout_ref.rollout)
        audit = self.output / "continuations.jsonl"
        for start in range(0, len(requests), self.pppo.request_chunk_size):
            chunk = requests[start:start + self.pppo.request_chunk_size]
            generation_started = time.perf_counter()
            generations = asyncio.run(generate_chunk(chunk, self.llm_server_manager.get_client(), sampling,
                                      self.pppo.max_concurrent_requests, self.pppo.request_timeout_seconds))
            timing_raw["pppo_continuation_generate"] = timing_raw.get("pppo_continuation_generate", 0) + time.perf_counter() - generation_started
            for score_start in range(0, len(generations), self.pppo.scoring_chunk_size):
                score_chunk = generations[score_start:score_start + self.pppo.scoring_chunk_size]
                fresh = scoring_batch(candidate, score_chunk, self.tokenizer.pad_token_id)
                scoring_started = time.perf_counter()
                normalized = self._score_batch_with_existing_reward_pipeline(fresh)
                values = correctness(normalized.extra_info, len(fresh))
                timing_raw["pppo_continuation_reward"] = timing_raw.get("pppo_continuation_reward", 0) + time.perf_counter() - scoring_started
                with audit.open("a") as f:
                    for generation, acc in zip(score_chunk, values, strict=True):
                        request = generation.request
                        if request.request_id in seen:
                            raise ValueError("duplicate PPPO branch scoring")
                        seen.add(request.request_id)
                        rewards[request.parent] += acc
                        counts[request.parent] += 1
                        tail_tokens += len(generation.tail)
                        input_tokens += len(request.prompt) + len(request.prefix)
                        f.write(json.dumps({"step": self.global_steps, "candidate": generation_batch_index,
                                "request_id": request.request_id, "parent": request.occurrence, "branch": request.branch,
                                "policy_version": version, "eta": self.prefix_schedule.eta, "seed": request.seed,
                                "acc": float(acc), "tail_tokens": len(generation.tail),
                                "tail_hash": hashlib.sha256(json.dumps(generation.tail).encode()).hexdigest(),
                                "stop_reason": generation.stop_reason, "finish_reason": generation.finish_reason}) + "\n")
                del fresh, normalized
            del generations
        if not np.all(counts == self.pppo.extra_continuations_per_prefix):
            raise ValueError("PPPO incomplete continuation count")
        reward = torch.as_tensor(rewards, dtype=torch.float32, device=lengths.device)
        a, total, weight = group_statistics(reward, lengths, candidate.non_tensor_batch["uid"])
        for key, value in {"pppo_reward": reward, "pppo_advantage": a,
                           "pppo_group_total_length": total, "pppo_row_weight": weight}.items():
            candidate.batch[key] = value
        candidate.non_tensor_batch["pppo_filter_reward"] = rewards
        for name, value in {"candidate_original_rows": len(candidate), "continuation_rows": len(requests),
                            "original_tokens": int(lengths.sum()), "tail_tokens": tail_tokens,
                            "continuation_prefill_tokens": input_tokens}.items():
            metrics[f"pppo/{name}"] = metrics.get(f"pppo/{name}", 0) + value
        metrics["pppo/eta_percent"] = self.prefix_schedule.eta
        metrics["pppo/candidate_original_accuracy"] = float(original_acc.mean())
        metrics["pppo/candidate_reward_mean"] = float(rewards.mean())
        with (self.output / "originals.jsonl").open("a") as f:
            for i in range(len(candidate)):
                tokens = candidate.batch["responses"][i, :int(lengths[i])].tolist()
                f.write(json.dumps({"step": self.global_steps, "parent": str(candidate.non_tensor_batch["trajectory_id"][i]),
                        "policy_version": version, "eta": self.prefix_schedule.eta, "length": int(lengths[i]),
                        "prefix_length": int(h[i]), "acc": float(original_acc[i]), "reward": float(rewards[i]),
                        "advantage": float(a[i]), "row_weight": float(weight[i]),
                        "token_hash": hashlib.sha256(json.dumps(tokens).encode()).hexdigest()}) + "\n")
        timing_raw["pppo_continuation_and_score"] = timing_raw.get("pppo_continuation_and_score", 0) + time.perf_counter()-started
        return candidate

    def _prepare_final_retained_batch(self, batch, metrics, timing_raw):
        self._validate_rollout_policy_version(batch)
        # Assert group integrity again after filtering/concat, before any row shuffling.
        a, total, weight = group_statistics(batch.batch["pppo_reward"], batch.batch["pppo_original_length"],
                                           batch.non_tensor_batch["uid"])
        for key, value in (("pppo_advantage", a), ("pppo_group_total_length", total), ("pppo_row_weight", weight)):
            if not torch.equal(batch.batch[key], value):
                raise ValueError(f"retained PPPO fields changed: {key}")
        batch = actor_batch(batch, self.pppo.actor_view, self.tokenizer.pad_token_id)
        metrics["pppo/actor_rows"] = len(batch)
        metrics["pppo/optimized_tokens"] = int(batch.batch["pppo_loss_mask"].sum())
        metrics["pppo/actor_input_tokens"] = int(batch.batch["attention_mask"].sum())
        metrics["pppo/zero_prefix_fallback"] = int(batch.meta_info["pppo_zero_prefix_fallback"])
        metrics["pppo/zero_advantage_fraction"] = float((a == 0).float().mean())
        self.checkpoint_manager.sleep_replicas()
        return batch

    def _compute_advantage_and_actor_update(self, batch, metrics, timing_raw):
        started = time.perf_counter()
        if self.pppo.get("optimizer_audit", {}).get("enabled", False):
            batch.meta_info["pppo_audit_dir"] = str(self.output / "optimizer_audit")
        else:
            batch.meta_info.pop("pppo_audit_dir", None)
        batch.meta_info["pppo_strict_update"] = True
        batch.meta_info["pppo_outer_step"] = self.global_steps
        output = self._update_actor(batch)
        expected_updates = (self.config.data.train_batch_size // self.config.actor_rollout_ref.actor.ppo_mini_batch_size
                            * self.config.actor_rollout_ref.actor.ppo_epochs)
        for key, expected in (("actor/pppo_optimizer_updates_this_outer", expected_updates),
                              ("actor/pppo_optimizer_updates_total", expected_updates * self.global_steps)):
            values = np.asarray(output.meta_info["metrics"].get(key, []), dtype=float)
            if values.size == 0 or not np.all(values == expected):
                raise RuntimeError(f"PPPO actual optimizer count mismatch: {key}={values}, expected {expected}")
        timing_raw["update_actor"] = time.perf_counter() - started
        return batch, output

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        result = super()._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)
        values = correctness(reward_extra_infos_dict, len(sample_uids))
        result["pppo/holdout_accuracy"] = float(values.mean())
        return result

    def _validate(self, *args, **kwargs):
        result = super()._validate(*args, **kwargs)
        s = self.pppo.schedule
        schedule_due = self.global_steps == 0 or self.global_steps % self.config.trainer.test_freq == 0
        changed = schedule_due and self.prefix_schedule.observe(self.global_steps, result["pppo/holdout_accuracy"],
                                               kind=s.type, patience=s.patience, min_delta=s.min_delta,
                                               cooldown_validations=s.cooldown_validations)
        self._last_validation = self.global_steps
        with (self.output / "schedule_events.jsonl").open("a") as f:
            f.write(json.dumps({"step": self.global_steps, "accuracy": result["pppo/holdout_accuracy"],
                                "changed": changed, **self.prefix_schedule.state_dict()}) + "\n")
        if self.pending_checkpoint is not None:
            self._commit_checkpoint()
        return result

    def _checkpoint_state(self, status):
        return {"status": status, "step": self.global_steps, "policy_version": self.global_steps,
                "config_hash": self.config_hash, "holdout_hash": self.pppo.holdout_hash,
                "schedule": self.prefix_schedule.state_dict(), "occurrences": self.occurrences,
                "data_epoch": self.data_epoch, "epoch_batches": self.epoch_batches}

    def _save_checkpoint(self):
        directory = Path(self.config.trainer.default_local_dir) / f"global_step_{self.global_steps}"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "pppo_state.json", self._checkpoint_state("model_save_pending"))
        super()._save_checkpoint()
        torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()},
                   directory / "pppo_driver_rng.pt")
        pending = self.config.trainer.test_freq > 0 and self.global_steps % self.config.trainer.test_freq == 0
        atomic_json(directory / "pppo_state.json", self._checkpoint_state("validation_pending" if pending else "step_pending"))
        self.pending_checkpoint = directory

    def _commit_checkpoint(self):
        if self.pending_checkpoint is None:
            return
        directory = self.pending_checkpoint
        required = [directory / "data.pt", directory / "pppo_driver_rng.pt"]
        for prefix in ("model", "optim", "extra_state"):
            expected = self.config.trainer.nnodes * self.config.trainer.n_gpus_per_node
            shards = [directory / "actor" / f"{prefix}_world_size_{expected}_rank_{rank}.pt"
                      for rank in range(expected)]
            actual = set((directory / "actor").glob(prefix + "_world_size_*_rank_*.pt"))
            if actual != set(shards):
                raise ValueError(f"incomplete {prefix} checkpoint shards")
            required.extend(shards)
        if any(not p.is_file() or p.stat().st_size == 0 for p in required):
            raise ValueError("incomplete PPPO checkpoint")
        state = self._checkpoint_state("committed")
        state["files"] = {str(p.relative_to(directory)): p.stat().st_size for p in required}
        atomic_json(directory / "pppo_state.json", state)
        atomic_json(Path(self.config.trainer.default_local_dir) / "pppo_latest.json", {"path": str(directory)})
        self.pending_checkpoint = None

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return
        directory = self.config.trainer.resume_from_path
        if self.config.trainer.resume_mode == "auto":
            latest = Path(self.config.trainer.default_local_dir) / "pppo_latest.json"
            if not latest.exists():
                raise ValueError("no committed PPPO checkpoint for auto resume")
            directory = json.loads(latest.read_text())["path"]
        directory = Path(directory)
        state = json.loads((directory / "pppo_state.json").read_text())
        if state["config_hash"] != self.config_hash or state["holdout_hash"] != self.pppo.holdout_hash:
            raise ValueError("PPPO resume configuration/holdout mismatch")
        if state["step"] != int(directory.name.split("_")[-1]):
            raise ValueError("PPPO checkpoint step mismatch")
        for name, size in state.get("files", {}).items():
            if (directory / name).stat().st_size != size:
                raise ValueError("PPPO checkpoint file size mismatch")
        if state["status"] not in ("committed", "validation_pending", "step_pending"):
            raise ValueError("checkpoint model save did not complete; use prior committed checkpoint")
        self.global_steps = state["step"]
        self.actor_rollout_wg.load_checkpoint(str(directory / "actor"), del_local_after_load=False)
        # Never infer the data cursor from optimizer/outer steps: refill consumes variable batches.
        self.data_epoch, self.epoch_batches = state["data_epoch"], state["epoch_batches"]
        self.train_dataloader.load_state_dict(torch.load(directory / "data.pt", weights_only=False))
        self.occurrences = state["occurrences"]
        self.prefix_schedule = PrefixSchedule(**state["schedule"])
        rng = torch.load(directory / "pppo_driver_rng.pt", weights_only=False)
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if state["status"] != "committed":
            self.pending_checkpoint = directory
        print(f"PPPO_RESTORED step={self.global_steps} eta={self.prefix_schedule.eta} occurrences={self.occurrences}", flush=True)

    def _flush_step_profile(self, timing_raw, metrics):
        if self._profile_should_stop_after_step() and self._last_validation != self.global_steps:
            # Acceptance-only final evaluation is recorded but must not advance the plateau schedule.
            result = super()._validate()
            metrics.update(result)
        self._commit_checkpoint()
        metrics["pppo/next_eta_percent"] = self.prefix_schedule.eta
        if "train/generated_prompt_groups" in metrics:
            metrics["pppo/valid_excess_groups"] = (metrics["train/generated_prompt_groups"]
                                                  - metrics["train/filtered_prompt_groups"]
                                                  - metrics["train/retained_prompt_groups"])
        with (self.output / "training_metrics.jsonl").open("a") as stream:
            stream.write(json.dumps({"step": self.global_steps, "metrics": metrics,
                                     "timing_seconds": timing_raw}, default=float, allow_nan=False) + "\n")

    def _profile_should_stop_after_step(self):
        stop = self.pppo.stop_after_step
        return stop is not None and self.global_steps >= stop
