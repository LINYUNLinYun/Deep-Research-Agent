#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics/stats.py
================================================================================
统计显著性检验工具：bootstrap 置信区间、效应量、配对 t 检验。

适用于小样本消融实验和 head-to-head benchmark 的统计严谨性验证。
================================================================================
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def bootstrap_ci_paired(
    diffs: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = 0,
    alternative: str = "greater",
) -> dict[str, Any]:
    """
    配对差异的 bootstrap 置信区间。

    Args:
        diffs: 配对差异列表（如 full_score - no_adv_score）
        n_bootstrap: bootstrap 采样次数
        confidence: 置信水平
        seed: 随机种子。默认使用固定种子，保证消融结果可复现；传 ``None`` 可显式关闭固定种子。
        alternative: ``greater``（默认，A 优于 B）、``less`` 或 ``two-sided``。

    Returns:
        dict with mean_diff, ci_lower, ci_upper, p_value, significant
    """
    if not diffs:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False}

    if not 0 < confidence < 1:
        raise ValueError("confidence 必须在 (0, 1) 内")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative 必须为 greater、less 或 two-sided")

    diffs_arr = np.asarray(diffs, dtype=float)
    diffs_arr = diffs_arr[np.isfinite(diffs_arr)]
    if diffs_arr.size == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False, "n": 0}
    mean_diff = float(np.mean(diffs_arr))

    rng = np.random.default_rng(seed)
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap 必须为正数")
    # Percentile bootstrap CI estimates sampling uncertainty around the observed
    # mean.  The old implementation estimated p from this *uncentred* sample,
    # which is not a null distribution and can report overly optimistic p-values.
    sample_indices = rng.integers(0, len(diffs_arr), size=(n_bootstrap, len(diffs_arr)))
    boot_means = np.mean(diffs_arr[sample_indices], axis=1)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_means, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_means, (1 - alpha / 2) * 100))

    # Paired randomisation/sign-flip test under H0: mean difference == 0.
    # This preserves the pairing and gives a valid null distribution without
    # pretending that an uncentred bootstrap sample is a hypothesis test.
    if len(diffs_arr) <= 16 and n_bootstrap >= (1 << len(diffs_arr)):
        signs = np.array(
            [
                [1.0 if mask & (1 << i) else -1.0 for i in range(len(diffs_arr))]
                for mask in range(1 << len(diffs_arr))
            ]
        )
        null_means = np.mean(signs * diffs_arr, axis=1)
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_bootstrap, len(diffs_arr)))
        null_means = np.mean(signs * diffs_arr, axis=1)

    if alternative == "greater":
        p_value = (float(np.sum(null_means >= mean_diff)) + 1.0) / (len(null_means) + 1.0)
        significant = ci_lower > 0
    elif alternative == "less":
        p_value = (float(np.sum(null_means <= mean_diff)) + 1.0) / (len(null_means) + 1.0)
        significant = ci_upper < 0
    else:
        p_value = (float(np.sum(np.abs(null_means) >= abs(mean_diff))) + 1.0) / (len(null_means) + 1.0)
        significant = ci_lower > 0 or ci_upper < 0

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "significant": significant,
        "n": int(len(diffs_arr)),
        "alternative": alternative,
        "seed": seed,
    }


def bootstrap_ci_two_sample(
    scores_a: list[float],
    scores_b: list[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = 0,
    alternative: str = "greater",
) -> dict[str, Any]:
    """
    两组独立样本的 bootstrap 置信区间（非配对）。

    Args:
        scores_a: 系统 A 的分数列表
        scores_b: 系统 B 的分数列表
    """
    if not scores_a or not scores_b:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False}

    if not 0 < confidence < 1:
        raise ValueError("confidence 必须在 (0, 1) 内")
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative 必须为 greater、less 或 two-sided")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap 必须为正数")

    a_arr = np.asarray(scores_a, dtype=float)
    b_arr = np.asarray(scores_b, dtype=float)
    a_arr = a_arr[np.isfinite(a_arr)]
    b_arr = b_arr[np.isfinite(b_arr)]
    if a_arr.size == 0 or b_arr.size == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0, "significant": False}
    mean_diff = float(np.mean(a_arr) - np.mean(b_arr))

    rng = np.random.default_rng(seed)
    a_idx = rng.integers(0, len(a_arr), size=(n_bootstrap, len(a_arr)))
    b_idx = rng.integers(0, len(b_arr), size=(n_bootstrap, len(b_arr)))
    boot_diffs = np.mean(a_arr[a_idx], axis=1) - np.mean(b_arr[b_idx], axis=1)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_diffs, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_diffs, (1 - alpha / 2) * 100))
    # Permutation test under the null that the two independent groups have the
    # same distribution.  Unlike the previous uncentred bootstrap p-value,
    # this explicitly constructs a null distribution by shuffling labels.
    pooled = np.concatenate([a_arr, b_arr])
    null_diffs = np.empty(n_bootstrap, dtype=float)
    n_a = len(a_arr)
    for i in range(n_bootstrap):
        perm = rng.permutation(pooled)
        null_diffs[i] = float(np.mean(perm[:n_a]) - np.mean(perm[n_a:]))
    if alternative == "greater":
        p_value = (float(np.sum(null_diffs >= mean_diff)) + 1.0) / (len(null_diffs) + 1.0)
        significant = ci_lower > 0
    elif alternative == "less":
        p_value = (float(np.sum(null_diffs <= mean_diff)) + 1.0) / (len(null_diffs) + 1.0)
        significant = ci_upper < 0
    else:
        p_value = (float(np.sum(np.abs(null_diffs) >= abs(mean_diff))) + 1.0) / (len(null_diffs) + 1.0)
        significant = ci_lower > 0 or ci_upper < 0

    return {
        "mean_diff": round(mean_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "significant": significant,
        "n_a": int(len(a_arr)),
        "n_b": int(len(b_arr)),
        "alternative": alternative,
        "seed": seed,
    }


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """计算 Cohen's d 效应量。"""
    if not scores_a or not scores_b:
        return 0.0
    a_arr = np.asarray(scores_a, dtype=float)
    b_arr = np.asarray(scores_b, dtype=float)
    if a_arr.size < 2 or b_arr.size < 2:
        return 0.0
    pooled_std = math.sqrt((np.var(a_arr, ddof=1) + np.var(b_arr, ddof=1)) / 2)
    if pooled_std < 1e-9:
        return 0.0
    return float((np.mean(a_arr) - np.mean(b_arr)) / pooled_std)


def paired_t_test(scores_a: list[float], scores_b: list[float]) -> dict[str, Any]:
    """配对 t 检验（假设正态分布）。作为 bootstrap 的补充。"""
    try:
        from scipy import stats
        diffs = np.array(scores_a) - np.array(scores_b)
        t_stat, p_value = stats.ttest_1samp(diffs, popmean=0)
        return {
            "t_statistic": round(float(t_stat), 4),
            "p_value": round(float(p_value), 4),
            "mean_diff": round(float(np.mean(diffs)), 4),
            "n": len(diffs),
        }
    except ImportError:
        # 无 scipy 时退化为 bootstrap
        return bootstrap_ci_paired([a - b for a, b in zip(scores_a, scores_b)])
