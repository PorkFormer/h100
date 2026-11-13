"""Metric semantics grounded in native accumulator, trainer and file logger."""
from reader import numeric

GROUPS = [
    ('01_reward', '训练奖励 / Training reward'), ('02_validation', '训练验证 / Online validation'),
    ('03_length', '回答长度 / Response length'), ('04_policy', '策略变化 / Policy change'),
    ('05_stability', '更新稳定性 / Update stability'), ('06_sampling', '动态采样 / Dynamic sampling'),
    ('07_trigger', 'NCBR触发 / NCBR trigger'), ('08_recovery', '续写收益 / Continuation outcomes'),
    ('09_unlock', '组解锁 / Group unlocking'), ('10_cost', '额外成本 / Extra cost'),
    ('11_speed', '训练速度 / Training speed'), ('12_resources', '资源与异常 / Resources and anomalies')]
B = 'boundary_return/'
DENOM = {
    'hit_cap_rate': [B+'candidate_count'],
    'long_success_rate_given_cap': [B+'valid_long_score_count'],
    'recovered_rate_given_cap_failure': [B+'transition_h_wrong_l_wrong_count', B+'transition_h_wrong_l_correct_count'],
    'regressed_rate_given_cap_success': [B+'transition_h_correct_l_correct_count', B+'transition_h_correct_l_wrong_count'],
    'unlocked_group_rate': [B+'short_locked_group_count'],
    'all_wrong_unlocked_group_rate': [B+'short_all_wrong_group_count'],
    'newly_locked_group_rate': [B+'short_unlocked_group_count'],
    'long_cap_rate_given_cap': [B+'hit_cap_count'],
    'extra_generated_token_ratio': ['train/generated_response_tokens'],
}
EXPECTED = ['critic/rewards/mean', 'val-core/<source>/acc/mean@1', 'response_length/mean',
            'actor/ppo_kl', 'actor/grad_norm', 'train/dynamic_sampling_accept_rate',
            B+'hit_cap_rate', B+'recovered_rate_given_cap_failure', B+'unlocked_group_rate',
            B+'extra_generated_tokens', 'timing_s/step', 'actor/perf/max_memory_allocated_gb']


def group(key):
    if key.startswith('val-'): return 1
    if key.startswith(('critic/score', 'critic/reward')): return 0
    if key.startswith(('response_length', 'prompt_length', 'num_turns')): return 2
    if key in ('actor/entropy', 'actor/ppo_kl', 'actor/kl_loss', 'actor/kl_coef'): return 3
    if key.startswith(('actor/perf/', 'global_seqlen', 'perf/mfu', 'response/')) or 'timeout' in key or 'drift' in key or 'preempted' in key: return 11
    if key.startswith(('actor/', 'critic/')): return 4
    if key.startswith('train/'): return 5
    if key.startswith(B):
        name = key[len(B):]
        if 'group' in name: return 8
        if any(s in name for s in ('recovered', 'regressed', 'transition_', 'task_score_delta', 'long_success')): return 7
        if name in ('candidate_count', 'hit_cap_count', 'hit_cap_rate', 'valid_long_score_count', 'continuation_request_count', 'long_cap_count', 'long_cap_rate_given_cap'): return 6
        return 9
    if key.startswith(('timing_', 'perf/throughput', 'perf/time_per_step', 'perf/total_num_tokens', 'training/')): return 10
    return 11


def spec(key):
    g = group(key)
    unit = 'native scalar'
    if 'memory' in key and key.endswith('_gb'): unit = 'GB (native worker aggregation)'
    elif 'num_preempted' in key: unit = 'count (negative sentinel unavailable)'
    elif key.startswith('timing_s/') and key.endswith(('/prompt_length', '/response_length')): unit = 'tokens'
    elif key == 'actor/entropy': unit = 'nats/token'
    elif key == 'actor/ppo_kl': unit = 'approximate KL'
    elif key.startswith(('training/', 'num_turns/')) or key.endswith(('_rows', '_chunk_count')): unit = 'count'
    elif key.startswith('timing_per_token_ms'): unit = 'ms/token'
    elif key.startswith('timing_s/') or 'seconds' in key or key == 'perf/time_per_step': unit = 's'
    elif any(s in key for s in ('rate', 'ratio', 'clipfrac', '/acc/', 'perf/mfu')): unit = 'fraction'
    elif 'tokens' in key or ('length' in key and 'ratio' not in key): unit = 'tokens'
    elif 'count' in key or 'groups' in key or 'trajectories' in key or 'num_gen_batches' in key: unit = 'count'
    elif key == 'perf/throughput': unit = 'tokens/s/GPU (native aggregate)'
    notes = ['Raw native metric; no interpolation.']
    if g == 0: notes += ['Retained training batch reward/score, not accuracy.']
    if g == 1: notes += ['Online validation; verify greedy settings in run config; not offline Avg@32/pass@32. Source name unchanged.']
    if g == 2: notes += ['Retained batch lengths; not all sampling candidates.']
    if g in (5, 6, 7, 8, 9): notes += ['Candidate pool before dynamic sampling retention; native acceptance rate is not retained/generated.']
    if key == 'actor/ppo_kl': notes += ['PPO approximate old/new policy KL, not reference-model KL.']
    if unit == 's': notes += ['Stages may nest/overlap; do not sum as total step time.']
    if 'memory' in key: notes += ['Preserve worker-reduced scope; not per-device memory.']
    if 'mfu' in key: notes += ['MFU is not measured GPU utilization.']
    name = key.removeprefix(B)
    denom = DENOM.get(name) if key.startswith(B) else None
    if key.startswith(B+'transition_') and key.endswith('_rate'): denom = [B+'valid_long_score_count']
    if key.startswith(B+'task_score_delta_') or key.startswith(B+'tail_tokens_'): denom = [B+'hit_cap_count']
    if key.startswith(B+'continuation_input_tokens_'): denom = [B+'continuation_request_count']
    if key == 'train/dynamic_sampling_accept_rate': denom = ['train/generated_prompt_groups']
    return dict(metric=key, group=GROUPS[g][0], unit=unit, notes=' '.join(notes), denominator=denom)


def value(data, key):
    raw = data.get(key)
    if key not in data: return None, '未记录 / not recorded'
    if not numeric(raw): return None, '无效数值 / invalid numeric'
    denom = spec(key)['denominator']
    if denom:
        counts = [data.get(k) for k in denom]
        # Old logs omit short_unlocked_group_count; infer only from explicit candidate group counts.
        if key == B+'newly_locked_group_rate' and counts == [None]:
            total, locked = data.get('train/generated_prompt_groups'), data.get(B+'short_locked_group_count')
            counts = [total - locked] if numeric(total) and numeric(locked) else [None]
        if not all(numeric(c) and c >= 0 for c in counts) or sum(counts) <= 0:
            return None, '分母缺失或为零 / denominator unavailable'
    if raw == 0 and any(s in key for s in ('queue_seconds_union', 'prefill_engine_seconds_union', 'decode_engine_seconds_union')):
        return None, '采集状态未确认 / backend zero unverified'
    if 'num_preempted' in key and raw < 0:
        return None, '后端哨兵值 / backend sentinel'
    return raw, '已记录 / recorded'
