"""Experiment: Logit Distribution Analysis via Logprobs.

Uses OpenAI API's logprobs feature to extract the probability distribution
over score tokens (1-5) at the exact position where the model outputs its
score. Compares distributions with vs. without AI disclosure to determine
whether the disclosure shifts probability mass toward lower scores as an
implicit prior, before the model has even reasoned about content quality.

Covers ALL conditions:
 - Original 2x2 factorial (control, disclosure_only, demographic_only, both)
 - Position variants from Change 1 (disclosure at start, middle, end, parenthetical)

This goes beyond final score analysis by examining the model's full
probability landscape at the scoring moment.
"""
import asyncio
import json
import math
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI
from config import *
from data_prep import load_feedback_collection, sample_diverse_items, create_experimental_conditions
from run_position_experiment import create_position_conditions, POSITIONS

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

MAX_CONCURRENT = 5
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)
N_RUNS_LOGPROBS = 1  
SCORE_TOKENS = ["1", "2", "3", "4", "5"]


def build_evaluation_prompt(condition, calibration="none"):
    """Build the evaluation prompt (same as main experiment)."""
    score_rubric = "\n".join([
        f"Score {k}: {v}" for k, v in sorted(condition["score_descriptions"].items())
    ])

    if calibration == "blind":
        response_text = condition["original_response"]
    else:
        response_text = condition["response"]

    system_msg = ("You are an expert evaluator. Your task is to evaluate "
                  "the quality of a response based on a specific rubric.")

    if calibration == "fairness":
        system_msg += "\n\n" + FAIRNESS_INSTRUCTION
    elif calibration == "evidence_first":
        system_msg += "\n\n" + EVIDENCE_FIRST_INSTRUCTION

    user_msg = f"""### Task Description:
Evaluate the following response based on the given instruction and scoring rubric.

### Instruction:
{condition["instruction"]}

### Response to Evaluate:
{response_text}

### Evaluation Criteria:
{condition["criteria"]}

### Score Rubric:
{score_rubric}

### Your Evaluation:
Provide a brief justification, then give a score from 1 to 5 in the format: [RESULT] X
"""
    return system_msg, user_msg


def extract_score(text):
    """Extract score from evaluation response."""
    match = re.search(r'\[RESULT\]\s*(\d)', text)
    if match:
        return int(match.group(1))
    match = re.search(r'[Ss]core[:\s]+(\d)', text)
    if match:
        return int(match.group(1))
    match = re.search(r'\b([1-5])\b', text[-50:])
    if match:
        return int(match.group(1))
    return None


def extract_score_logprobs(logprobs_content):
    """Extract probability distribution over score tokens from logprobs.

    Scans the logprobs for the token following '[RESULT]' and extracts
    the probability distribution over score tokens 1-5.

    Args:
        logprobs_content: List of token logprob objects from the API response.

    Returns:
        dict with:
          - score_probs: {1: prob, 2: prob, ..., 5: prob} (normalized)
          - score_logprobs: {1: logprob, ..., 5: logprob} (raw log probabilities)
          - score_position: index in token sequence where score was found
          - top_tokens_at_score: all top tokens and their probs at score position
          - chosen_score: the actual score token chosen by the model
    """
    if not logprobs_content:
        return None

    score_position = None
    for i, token_info in enumerate(logprobs_content):
        token_text = token_info.token
        if i >= 1:
            prev_text = "".join(t.token for t in logprobs_content[max(0, i-5):i])
            if "[RESULT]" in prev_text or "RESULT]" in prev_text:
                stripped = token_text.strip()
                if stripped in SCORE_TOKENS:
                    score_position = i
                    break

    if score_position is None:
        min_pos = int(len(logprobs_content) * 0.5)
        for i in range(len(logprobs_content) - 1, min_pos, -1):
            token_text = logprobs_content[i].token.strip()
            if token_text in SCORE_TOKENS:
                score_position = i
                break

    if score_position is None:
        return None

    token_info = logprobs_content[score_position]
    chosen_token = token_info.token.strip()

    # extract top logprobs at the score position
    score_logprobs = {}
    score_probs = {}
    top_tokens_at_score = {}

    # token's own logprob
    if chosen_token in SCORE_TOKENS:
        score_logprobs[int(chosen_token)] = token_info.logprob
        score_probs[int(chosen_token)] = math.exp(token_info.logprob)

    # Top logprobs 
    if token_info.top_logprobs:
        for alt in token_info.top_logprobs:
            alt_token = alt.token.strip()
            top_tokens_at_score[alt_token] = {
                "logprob": alt.logprob,
                "prob": math.exp(alt.logprob),
            }
            if alt_token in SCORE_TOKENS:
                score_logprobs[int(alt_token)] = alt.logprob
                score_probs[int(alt_token)] = math.exp(alt.logprob)

    # normalize score probabilities to sum to 1 over observed score tokens
    total_prob = sum(score_probs.values())
    if total_prob > 0:
        normalized_probs = {k: v / total_prob for k, v in score_probs.items()}
    else:
        normalized_probs = score_probs

    return {
        "score_probs": score_probs,
        "score_probs_normalized": normalized_probs,
        "score_logprobs": score_logprobs,
        "score_position": score_position,
        "total_tokens": len(logprobs_content),
        "top_tokens_at_score": top_tokens_at_score,
        "chosen_score": int(chosen_token) if chosen_token in SCORE_TOKENS else None,
    }


async def call_evaluator_with_logprobs(system_msg, user_msg, run_id=0):
    """API call with logprobs enabled."""
    async with SEMAPHORE:
        for attempt in range(6):
            try:
                response = await aclient.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    seed=SEED + run_id,
                    logprobs=True,
                    top_logprobs=20,  
                )
                text = response.choices[0].message.content
                score = extract_score(text)

                logprobs_content = None
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    logprobs_content = response.choices[0].logprobs.content

                logprob_analysis = extract_score_logprobs(logprobs_content)

                return {
                    "raw_response": text,
                    "score": score,
                    "logprob_analysis": logprob_analysis,
                    "tokens_used": {
                        "prompt": response.usage.prompt_tokens,
                        "completion": response.usage.completion_tokens,
                    },
                }
            except Exception as e:
                if attempt < 5:
                    wait = min(2 ** (attempt + 1), 60)
                    await asyncio.sleep(wait)
                else:
                    return {
                        "raw_response": str(e),
                        "score": None,
                        "logprob_analysis": None,
                        "tokens_used": {},
                    }


async def evaluate_condition_logprobs(condition, calibration="none"):
    """Evaluate a single condition and return logprob analysis."""
    sys_msg, user_msg = build_evaluation_prompt(condition, calibration)
    result = await call_evaluator_with_logprobs(sys_msg, user_msg, run_id=0)

    return {
        "sample_id": condition["sample_id"],
        "condition": condition["condition"],
        "position": condition.get("position", "none"),
        "disclosure": condition.get("disclosure", False),
        "demographic": condition.get("demographic", False),
        "calibration": calibration,
        "ground_truth_score": condition["ground_truth_score"],
        "score": result["score"],
        "logprob_analysis": result["logprob_analysis"],
        "tokens_used": result["tokens_used"],
        "raw_response": result["raw_response"],
    }


async def run_logprobs_experiment(conditions, calibration="none"):
    """Run logprobs experiment for all conditions."""
    print(f"  [{calibration}] Starting {len(conditions)} conditions "
          f"(max {MAX_CONCURRENT} concurrent)...")
    t0 = time.time()

    tasks = [evaluate_condition_logprobs(c, calibration) for c in conditions]

    batch_size = 50
    all_results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        all_results.extend(batch_results)
        elapsed = time.time() - t0
        print(f"  [{calibration}] Progress: {min(i + batch_size, len(tasks))}/{len(tasks)} "
              f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    valid_count = sum(1 for r in all_results if r["logprob_analysis"] is not None)
    print(f"  [{calibration}] Completed in {elapsed:.0f}s "
          f"({valid_count}/{len(all_results)} with valid logprobs)")
    return all_results


def _compare_condition_vs_control(control_by_sample, condition_results, condition_label):
    """Helper: compare a set of condition results against control, paired by sample_id.

    Returns dict of analysis results or None if too few pairs.
    """
    import numpy as np
    from scipy import stats

    cond_by_sample = {}
    for r in condition_results:
        if r["logprob_analysis"] is not None:
            cond_by_sample[r["sample_id"]] = r

    common = set(control_by_sample.keys()) & set(cond_by_sample.keys())
    if len(common) < 5:
        return None

    #per score probability comparison
    score_stats = {}
    for s in range(1, 6):
        ctrl_vals = []
        cond_vals = []
        for sid in common:
            ctrl_norm = control_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
            cond_norm = cond_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
            ctrl_vals.append(ctrl_norm.get(s, 0.0))
            cond_vals.append(cond_norm.get(s, 0.0))
        ctrl_arr = np.array(ctrl_vals)
        cond_arr = np.array(cond_vals)
        diff = cond_arr - ctrl_arr
        if len(diff[diff != 0]) >= 5:
            t, p = stats.ttest_rel(cond_arr, ctrl_arr)
        else:
            t, p = 0, 1.0
        score_stats[s] = {
            "ctrl_mean": float(np.mean(ctrl_arr)),
            "cond_mean": float(np.mean(cond_arr)),
            "diff": float(np.mean(diff)),
            "p_value": float(p),
        }

    # expected score comparison
    ctrl_expected = []
    cond_expected = []
    for sid in common:
        ctrl_norm = control_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
        cond_norm = cond_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
        ctrl_expected.append(sum(s * ctrl_norm.get(s, 0) for s in range(1, 6)))
        cond_expected.append(sum(s * cond_norm.get(s, 0) for s in range(1, 6)))

    ctrl_exp_arr = np.array(ctrl_expected)
    cond_exp_arr = np.array(cond_expected)
    diff_exp = cond_exp_arr - ctrl_exp_arr
    t_exp, p_exp = stats.ttest_rel(cond_exp_arr, ctrl_exp_arr)
    d_exp = np.mean(diff_exp) / np.std(diff_exp, ddof=1) if np.std(diff_exp) > 0 else 0

    # entropy comparison
    ctrl_entropies = []
    cond_entropies = []
    for sid in common:
        ctrl_norm = control_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
        cond_norm = cond_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
        ctrl_entropies.append(-sum(p * math.log2(p) for p in ctrl_norm.values() if p > 0))
        cond_entropies.append(-sum(p * math.log2(p) for p in cond_norm.values() if p > 0))
    ctrl_h_arr = np.array(ctrl_entropies)
    cond_h_arr = np.array(cond_entropies)
    t_h, p_h = stats.ttest_rel(cond_h_arr, ctrl_h_arr)

    # KL divergence
    kl_divs = []
    eps = 1e-8
    for sid in common:
        ctrl_norm = control_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
        cond_norm = cond_by_sample[sid]["logprob_analysis"].get("score_probs_normalized", {})
        kl = sum(max(ctrl_norm.get(s, eps), eps) *
                 math.log(max(ctrl_norm.get(s, eps), eps) / max(cond_norm.get(s, eps), eps))
                 for s in range(1, 6))
        kl_divs.append(kl)
    kl_arr = np.array(kl_divs)

    return {
        "condition": condition_label,
        "n_pairs": len(common),
        "score_stats": score_stats,
        "expected_score": {
            "control_mean": float(np.mean(ctrl_exp_arr)),
            "condition_mean": float(np.mean(cond_exp_arr)),
            "diff": float(np.mean(diff_exp)),
            "t_stat": float(t_exp),
            "p_value": float(p_exp),
            "cohens_d": float(d_exp),
        },
        "entropy": {
            "control_mean": float(np.mean(ctrl_h_arr)),
            "condition_mean": float(np.mean(cond_h_arr)),
            "diff": float(np.mean(cond_h_arr - ctrl_h_arr)),
            "t_stat": float(t_h),
            "p_value": float(p_h),
        },
        "kl_divergence": {
            "mean": float(np.mean(kl_arr)),
            "median": float(np.median(kl_arr)),
            "std": float(np.std(kl_arr)),
        },
    }


def analyze_logprob_results(all_results):
    """Analyze logprob distributions for all conditions vs control."""
    import numpy as np
    from scipy import stats

    print("\n" + "=" * 70)
    print("LOGPROB DISTRIBUTION ANALYSIS")
    print("=" * 70)

    control_by_sample = {}
    for r in all_results:
        if r["logprob_analysis"] is None:
            continue
        if r["condition"] == "control":
            control_by_sample[r["sample_id"]] = r

    print(f"\nControl samples with valid logprobs: {len(control_by_sample)}")

    conditions_map = {}
    for r in all_results:
        if r["logprob_analysis"] is None:
            continue
        cond = r["condition"]
        if cond == "control":
            continue
        conditions_map.setdefault(cond, []).append(r)

    all_condition_names = sorted(conditions_map.keys())
    print(f"Conditions to analyze: {all_condition_names}")

    all_analyses = {}
    for cond_name in all_condition_names:
        cond_results = conditions_map[cond_name]
        analysis = _compare_condition_vs_control(control_by_sample, cond_results, cond_name)
        if analysis is None:
            print(f"\n  [{cond_name}] Too few valid pairs, skipping.")
            continue
        all_analyses[cond_name] = analysis

        exp = analysis["expected_score"]
        ent = analysis["entropy"]
        kl = analysis["kl_divergence"]
        sig = "***" if exp["p_value"] < 0.001 else "**" if exp["p_value"] < 0.01 else "*" if exp["p_value"] < 0.05 else "ns"

        print(f"\n{'─'*60}")
        print(f"  Condition: {cond_name} (n={analysis['n_pairs']} pairs)")
        print(f"{'─'*60}")

        print(f"  Score probability shifts (condition - control):")
        print(f"    {'Score':<8} {'Control':>10} {cond_name:>14} {'Diff':>10} {'p':>10}")
        for s in range(1, 6):
            ss = analysis["score_stats"][s]
            sig_s = "*" if ss["p_value"] < 0.05 else ""
            print(f"    {s:<8} {ss['ctrl_mean']:>10.4f} {ss['cond_mean']:>14.4f} "
                  f"{ss['diff']:>+10.4f} {ss['p_value']:>10.4f} {sig_s}")

        print(f"  Expected score: control={exp['control_mean']:.4f}, "
              f"{cond_name}={exp['condition_mean']:.4f}, "
              f"diff={exp['diff']:+.4f}, p={exp['p_value']:.4f}, d={exp['cohens_d']:.3f} {sig}")
        print(f"  Entropy: control={ent['control_mean']:.4f}, "
              f"{cond_name}={ent['condition_mean']:.4f}, "
              f"diff={ent['diff']:+.4f}, p={ent['p_value']:.4f}")
        print(f"  KL divergence: mean={kl['mean']:.4f}, median={kl['median']:.4f}")

    # summary table across all conditions
    print(f"\n{'='*70}")
    print("SUMMARY: EXPECTED SCORE SHIFT BY CONDITION")
    print(f"{'='*70}")
    print(f"{'Condition':<25} {'Ctrl':>7} {'Cond':>7} {'Diff':>8} {'p':>9} {'d':>7} {'Sig':>5}")
    print("-" * 70)
    for cond_name in all_condition_names:
        if cond_name not in all_analyses:
            continue
        exp = all_analyses[cond_name]["expected_score"]
        sig = "***" if exp["p_value"] < 0.001 else "**" if exp["p_value"] < 0.01 else "*" if exp["p_value"] < 0.05 else "ns"
        print(f"  {cond_name:<23} {exp['control_mean']:>7.3f} {exp['condition_mean']:>7.3f} "
              f"{exp['diff']:>+8.4f} {exp['p_value']:>9.4f} {exp['cohens_d']:>7.3f} {sig:>5}")

    # position specific comparison
    position_conditions = [c for c in all_condition_names if c.startswith("disclosure_")]
    if len(position_conditions) > 1:
        print(f"\n{'='*70}")
        print("POSITION COMPARISON: WHICH POSITION SHIFTS DISTRIBUTIONS MOST?")
        print(f"{'='*70}")
        print(f"{'Position':<25} {'E[score] shift':>15} {'KL div':>10} {'Entropy diff':>13}")
        print("-" * 65)
        for cond_name in position_conditions:
            if cond_name not in all_analyses:
                continue
            a = all_analyses[cond_name]
            print(f"  {cond_name:<23} {a['expected_score']['diff']:>+15.4f} "
                  f"{a['kl_divergence']['mean']:>10.4f} {a['entropy']['diff']:>+13.4f}")

    return all_analyses


async def main():
    print("=" * 70)
    print("LOGPROB DISTRIBUTION EXPERIMENT")
    print("(All conditions: 2x2 factorial + position variants)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {MODEL}")
    print()

    print("Loading dataset...")
    ds = load_feedback_collection()
    indices = sample_diverse_items(ds, N_SAMPLES)

    print("Creating 2x2 factorial conditions...")
    factorial_conditions = create_experimental_conditions(ds, indices)
    print(f"  {len(factorial_conditions)} factorial conditions")


    print("Creating position variant conditions...")
    position_conditions = create_position_conditions(ds, indices)
    print(f"  {len(position_conditions)} position conditions")

    # deduplicate controls: both sets have control conditions for the same samples
    # keep one control per sample_id (from factorial), then add all non-control conditions.
    seen_control_samples = set()
    all_conditions = []

    # add factorial conditions (control + disclosure_only + demographic_only + both)
    for c in factorial_conditions:
        if c["condition"] == "control":
            if c["sample_id"] not in seen_control_samples:
                seen_control_samples.add(c["sample_id"])
                all_conditions.append(c)
        else:
            all_conditions.append(c)

    for c in position_conditions:
        if c["condition"] != "control":
            all_conditions.append(c)

    n_control = sum(1 for c in all_conditions if c["condition"] == "control")
    n_factorial = sum(1 for c in all_conditions if c["condition"] in
                      ("disclosure_only", "demographic_only", "both"))
    n_position = sum(1 for c in all_conditions if c["condition"].startswith("disclosure_")
                     and c["condition"] != "disclosure_only")

    print(f"\nTotal conditions for logprobs:")
    print(f"  Control:          {n_control}")
    print(f"  2x2 factorial:    {n_factorial} (disclosure_only, demographic_only, both)")
    print(f"  Position variants:{n_position} (start, middle, end, parenthetical)")
    print(f"  TOTAL:            {len(all_conditions)} API calls")
    print()

    t0 = time.time()

    # run logprobs on everything
    results = await run_logprobs_experiment(all_conditions, calibration="none")
    total_time = time.time() - t0

    results_path = os.path.join(RESULTS_DIR, "logprobs_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {results_path}")

    all_analyses = analyze_logprob_results(results)

    condition_names = sorted(set(r["condition"] for r in results))
    metadata = {
        "experiment": "logprob_distribution",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "top_logprobs": 20,
        "conditions": condition_names,
        "total_conditions": len(all_conditions),
        "total_time_seconds": total_time,
    }
    meta_path = os.path.join(RESULTS_DIR, "logprobs_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if all_analyses:
        analysis_path = os.path.join(RESULTS_DIR, "logprobs_analysis_summary.json")
        with open(analysis_path, "w") as f:
            json.dump(all_analyses, f, indent=2, default=str)
        print(f"Saved analysis summary to {analysis_path}")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total API calls: {len(all_conditions)}")


if __name__ == "__main__":
    asyncio.run(main())
