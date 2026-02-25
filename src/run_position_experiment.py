"""Experiment: Disclosure Position Effect.
We already have discovered from the original experiment that AI 
disclosure leads to a penalty from the judge model.
Here we test whether the position of AI disclosure text affects the evaluator's
score penalty. Positions tested:
  - start: prepended at the beginning (original method)
  - middle: inserted at the approximate midpoint of the response
  - end: appended at the end of the response
  - parenthetical: embedded as a parenthetical note mid-sentence

Using the same 100 samples from the original experiment to enable
direct comparison with baseline results.
"""
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import AsyncOpenAI
from config import *
from data_prep import load_feedback_collection, sample_diverse_items

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY)

MAX_CONCURRENT = 5
SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT)
N_RUNS_POS = 3

DISCLOSURE_TEXT_PLAIN = "Note: This response was written with AI assistance."
DISCLOSURE_PARENTHETICAL = "(written with AI assistance)"

# we are testing these positions
POSITIONS = ["start", "middle", "end", "parenthetical"]


def insert_disclosure_at_position(response_text, position):
    """Insert the AI disclosure at the specified position in the response.

    Args:
        response_text: The original response text.
        position: One of 'start', 'middle', 'end', 'parenthetical'.

    Returns:
        Modified response text with disclosure inserted.
    """
    if position == "start":
        return DISCLOSURE_TEXT_PLAIN + "\n\n" + response_text

    elif position == "end":
        return response_text + "\n\n" + DISCLOSURE_TEXT_PLAIN

    elif position == "middle":
        # split into sentences and insert at the midpoint
        sentences = re.split(r'(?<=[.!?])\s+', response_text)
        if len(sentences) <= 2:
            # short response: insert between the two halves
            mid = len(response_text) // 2
            # finding nearest space to avoid splitting a word
            space_idx = response_text.find(' ', mid)
            if space_idx == -1:
                space_idx = mid
            return (response_text[:space_idx] + " " +
                    DISCLOSURE_TEXT_PLAIN + " " +
                    response_text[space_idx:])
        else:
            mid_idx = len(sentences) // 2
            before = ' '.join(sentences[:mid_idx])
            after = ' '.join(sentences[mid_idx:])
            return before + " " + DISCLOSURE_TEXT_PLAIN + " " + after

    elif position == "parenthetical":
        # insert parenthetical after the first sentence and punctuation
        match = re.search(r'(?<=[.!?])\s+', response_text)
        if match:
            insert_pos = match.start() + 1  
            return (response_text[:insert_pos] + " " +
                    DISCLOSURE_PARENTHETICAL + " " +
                    response_text[insert_pos:])
        else:
            # no sentence boundary found so prepend
            return DISCLOSURE_PARENTHETICAL + " " + response_text

    else:
        raise ValueError(f"Unknown position: {position}")


def create_position_conditions(ds, indices):
    """Create experimental conditions for each disclosure position.

    For each sample, creates:
      - 1 control condition (no disclosure)
      - 4 position conditions (start, middle, end, parenthetical)
    Total: 5 conditions per sample = 500 conditions for 100 samples.
    """
    conditions = []
    for idx in indices:
        item = ds[idx]
        base_instruction = item["orig_instruction"]
        base_response = item["orig_response"]
        base_criteria = item["orig_criteria"]
        base_score = int(item["orig_score"])
        score_descriptions = {
            str(k): item[f"orig_score{k}_description"] for k in range(1, 6)
        }

        # control
        conditions.append({
            "sample_id": idx,
            "condition": "control",
            "position": "none",
            "instruction": base_instruction,
            "response": base_response,
            "original_response": base_response,
            "criteria": base_criteria,
            "ground_truth_score": base_score,
            "score_descriptions": score_descriptions,
        })

        # 1 condition per disclosure position
        for position in POSITIONS:
            modified = insert_disclosure_at_position(base_response, position)
            conditions.append({
                "sample_id": idx,
                "condition": f"disclosure_{position}",
                "position": position,
                "instruction": base_instruction,
                "response": modified,
                "original_response": base_response,
                "criteria": base_criteria,
                "ground_truth_score": base_score,
                "score_descriptions": score_descriptions,
            })

    return conditions


def build_evaluation_prompt(condition):
    """Build the evaluation prompt (vanilla, no calibration)."""
    score_rubric = "\n".join([
        f"Score {k}: {v}" for k, v in sorted(condition["score_descriptions"].items())
    ])
    response_text = condition["response"]

    system_msg = ("You are an expert evaluator. Your task is to evaluate "
                  "the quality of a response based on a specific rubric.")

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


async def call_evaluator_async(system_msg, user_msg, run_id=0):
    """Async API call with semaphore for rate limiting."""
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
                )
                text = response.choices[0].message.content
                score = extract_score(text)
                return {
                    "raw_response": text,
                    "score": score,
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
                    return {"raw_response": str(e), "score": None, "tokens_used": {}}


async def evaluate_single_condition(condition, n_runs):
    """Evaluate a single condition with multiple runs."""
    sys_msg, user_msg = build_evaluation_prompt(condition)
    tasks = [call_evaluator_async(sys_msg, user_msg, run_id) for run_id in range(n_runs)]
    results = await asyncio.gather(*tasks)

    run_scores = [r["score"] for r in results]
    raw_responses = [r["raw_response"] for r in results]
    total_tokens = {"prompt": 0, "completion": 0}
    for r in results:
        t = r.get("tokens_used", {})
        total_tokens["prompt"] += t.get("prompt", 0)
        total_tokens["completion"] += t.get("completion", 0)

    valid_scores = [s for s in run_scores if s is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else None

    return {
        "sample_id": condition["sample_id"],
        "condition": condition["condition"],
        "position": condition["position"],
        "ground_truth_score": condition["ground_truth_score"],
        "run_scores": run_scores,
        "mean_score": mean_score,
        "n_valid_runs": len(valid_scores),
        "tokens_used": total_tokens,
        "raw_responses": raw_responses,
    }


async def run_position_experiment(conditions, n_runs=N_RUNS_POS):
    """Run all position conditions concurrently."""
    print(f"Starting {len(conditions)} conditions x {n_runs} runs "
          f"({len(conditions) * n_runs} API calls, max {MAX_CONCURRENT} concurrent)...")
    t0 = time.time()

    tasks = [evaluate_single_condition(c, n_runs) for c in conditions]

    batch_size = 50
    all_results = []
    for i in range(0, len(tasks), batch_size):
        batch = tasks[i:i + batch_size]
        batch_results = await asyncio.gather(*batch)
        all_results.extend(batch_results)
        elapsed = time.time() - t0
        print(f"  Progress: {min(i + batch_size, len(tasks))}/{len(tasks)} "
              f"({elapsed:.0f}s elapsed)")

    elapsed = time.time() - t0
    valid_count = sum(1 for r in all_results if r["mean_score"] is not None)
    print(f"Completed in {elapsed:.0f}s ({valid_count}/{len(all_results)} valid)")
    return all_results


def print_summary(results):
    """Print summary of position experiment results."""
    import numpy as np
    from scipy import stats

    print("\n" + "=" * 60)
    print("POSITION EXPERIMENT RESULTS")
    print("=" * 60)

    # group by position
    scores_by_pos = {}
    for r in results:
        if r["mean_score"] is not None:
            scores_by_pos.setdefault(r["position"], []).append(r)

    # control mean
    ctrl_by_sample = {}
    for r in scores_by_pos.get("none", []):
        ctrl_by_sample[r["sample_id"]] = r["mean_score"]
    ctrl_mean = np.mean(list(ctrl_by_sample.values()))

    print(f"\nControl mean score: {ctrl_mean:.3f} (n={len(ctrl_by_sample)})")
    print(f"\n{'Position':<18} {'Mean':>7} {'Penalty':>9} {'t-stat':>8} {'p-value':>9} {'Cohen d':>9} {'Sig':>5}")
    print("-" * 70)

    for pos in POSITIONS:
        pos_results = scores_by_pos.get(pos, [])
        # pairing with control by sample_id
        paired_diffs = []
        ctrl_vals = []
        pos_vals = []
        for r in pos_results:
            if r["sample_id"] in ctrl_by_sample:
                c = ctrl_by_sample[r["sample_id"]]
                p = r["mean_score"]
                paired_diffs.append(p - c)
                ctrl_vals.append(c)
                pos_vals.append(p)

        if len(paired_diffs) >= 5:
            diffs = np.array(paired_diffs)
            mean_penalty = np.mean(diffs)
            t_stat, p_val = stats.ttest_rel(pos_vals, ctrl_vals)
            d = np.mean(diffs) / np.std(diffs, ddof=1) if np.std(diffs) > 0 else 0
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            pos_mean = np.mean(pos_vals)
            print(f"  {pos:<16} {pos_mean:>7.3f} {mean_penalty:>+9.3f} {t_stat:>8.2f} {p_val:>9.4f} {d:>9.2f} {sig:>5}")

    # pairwise comparisons between positions
    print(f"\n{'Pairwise Position Comparisons (paired t-test)':}")
    print("-" * 50)
    for i, pos_a in enumerate(POSITIONS):
        for pos_b in POSITIONS[i+1:]:
            a_by_sample = {r["sample_id"]: r["mean_score"]
                          for r in scores_by_pos.get(pos_a, []) if r["mean_score"] is not None}
            b_by_sample = {r["sample_id"]: r["mean_score"]
                          for r in scores_by_pos.get(pos_b, []) if r["mean_score"] is not None}
            common = set(a_by_sample) & set(b_by_sample)
            if len(common) >= 5:
                a_vals = [a_by_sample[s] for s in common]
                b_vals = [b_by_sample[s] for s in common]
                t, p = stats.ttest_rel(a_vals, b_vals)
                diff = np.mean(np.array(a_vals) - np.array(b_vals))
                sig = "*" if p < 0.05 else "ns"
                print(f"  {pos_a} vs {pos_b}: diff={diff:+.3f}, t={t:.2f}, p={p:.4f} {sig}")


async def main():
    print("=" * 70)
    print("DISCLOSURE POSITION EXPERIMENT")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Model: {MODEL}")
    print(f"Positions: {POSITIONS}")
    print(f"N_SAMPLES: {N_SAMPLES}, N_RUNS: {N_RUNS_POS}")
    print()
    print("Loading dataset")
    ds = load_feedback_collection()
    indices = sample_diverse_items(ds, N_SAMPLES)
    print(f"Sampled {len(indices)} items")

    print("Creating position conditions...")
    conditions = create_position_conditions(ds, indices)
    print(f"Created {len(conditions)} conditions "
          f"(100 control + {len(POSITIONS)} positions x 100 samples)")
    print()

    cond_path = os.path.join(RESULTS_DIR, "position_conditions.json")
    with open(cond_path, "w") as f:
        json.dump(conditions, f, indent=2)
    print(f"Saved conditions to {cond_path}")

    #running
    t0 = time.time()
    results = await run_position_experiment(conditions, n_runs=N_RUNS_POS)
    total_time = time.time() - t0

    results_path = os.path.join(RESULTS_DIR, "position_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {results_path}")

    print_summary(results)

    metadata = {
        "experiment": "disclosure_position",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL,
        "n_samples": N_SAMPLES,
        "n_runs": N_RUNS_POS,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "positions": POSITIONS,
        "total_conditions": len(conditions),
        "total_time_seconds": total_time,
        "total_api_calls": len(conditions) * N_RUNS_POS,
    }
    meta_path = os.path.join(RESULTS_DIR, "position_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nSaved metadata to {meta_path}")

    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Total API calls: {len(conditions) * N_RUNS_POS}")


if __name__ == "__main__":
    asyncio.run(main())
