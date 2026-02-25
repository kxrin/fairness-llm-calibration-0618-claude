"""Analysis: Reasoning Linguistic Comparison.

Compares the evaluator's reasoning text between disclosure and control
conditions for the same response. Identifies linguistic signatures of
implicit bias by finding words and criticisms that appear more
frequently in disclosure-condition reasoning than in control reasoning.

This reveals whether the model manufactures quality criticisms to justify
a score it was already biased toward, rather than reasoning from content.
"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, PLOTS_DIR, SEED

np.random.seed(SEED)

# negative-tone  words to track
NEGATIVE_MARKERS = [
    "generic", "vague", "superficial", "lacks", "lacking", "limited",
    "basic", "simplistic", "shallow", "insufficient", "weak", "poor",
    "unclear", "incomplete", "inadequate", "mediocre", "repetitive",
    "formulaic", "boilerplate", "template", "cliché", "cliche",
    "unoriginal", "derivative", "mundane", "ordinary", "unremarkable",
    "fails", "missing", "absent", "deficient", "flawed",
]
# positive tone words to track
POSITIVE_MARKERS = [
    "excellent", "strong", "thorough", "comprehensive", "detailed",
    "insightful", "nuanced", "sophisticated", "effective", "well",
    "clear", "precise", "accurate", "compelling", "thoughtful",
    "creative", "original", "innovative", "impressive", "exemplary",
    "exceptional", "outstanding", "robust", "solid", "convincing",
]

# AI related terms
AI_MARKERS = [
    "ai", "artificial", "generated", "automated", "machine",
    "assistance", "assisted", "tool", "bot", "algorithm",
]


def load_baseline_results():
    path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(path) as f:
        return json.load(f)


def get_paired_reasoning(results):
    """
    returns dict mapping sample_id -> {
        "control": [list of reasoning texts],
        "disclosure": [list of reasoning texts],
        "ground_truth_score": int,
        "control_scores": [...],
        "disclosure_scores": [...],
    }
    """
    pairs = {}
    for r in results:
        sid = r["sample_id"]
        if sid not in pairs:
            pairs[sid] = {
                "control": [],
                "disclosure": [],
                "ground_truth_score": r["ground_truth_score"],
                "control_scores": [],
                "disclosure_scores": [],
            }
        if r["condition"] == "control":
            pairs[sid]["control"].extend(r["raw_responses"])
            pairs[sid]["control_scores"].extend(
                s for s in r["run_scores"] if s is not None)
        elif r["condition"] == "disclosure_only":
            pairs[sid]["disclosure"].extend(r["raw_responses"])
            pairs[sid]["disclosure_scores"].extend(
                s for s in r["run_scores"] if s is not None)

    return {sid: data for sid, data in pairs.items()
            if data["control"] and data["disclosure"]}


def tokenize_simple(text):
    """basic tokenization: lowercase, split on non-alpha"""
    if not text:
        return []
    words = re.findall(r'[a-z]+', text.lower())
    return [w for w in words if len(w) >= 2]


def count_markers(texts, markers):
    """count occurrences of marker words in a list of texts.

    returns total count and per-text rate.
    """
    totl_words = 0
    marker_counts = Counter()
    for text in texts:
        words = tokenize_simple(text)
        totl_words += len(words)
        for word in words:
            if word in markers:
                marker_counts[word] += 1
    return marker_counts, totl_words


def compute_word_frequencies(texts):
    """calculate the word frequency distribution from a list of texts"""
    wc = Counter()
    total = 0
    for text in texts:
        words = tokenize_simple(text)
        wc.update(words)
        total += len(words)
    return wc, total


def run_reasoning_analysis():
    """ analysis: compare reasoning between conditions"""
    print("=" * 70)
    print("REASONING LINGUISTIC ANALYSIS")
    print("=" * 70)

    results = load_baseline_results()
    pairs = get_paired_reasoning(results)
    print(f"Paired samples (control + disclosure): {len(pairs)}")

    # aggregate marker word analysis 
    print("\n" + "=" * 60)
    print("1. NEGATIVE-TONE MARKER FREQUENCY COMPARISON")
    print("=" * 60)

    all_ctrl_texts = []
    all_disc_texts = []
    for data in pairs.values():
        all_ctrl_texts.extend(data["control"])
        all_disc_texts.extend(data["disclosure"])

    ctrl_neg_counts, ctrl_total_words = count_markers(all_ctrl_texts, NEGATIVE_MARKERS)
    disc_neg_counts, disc_total_words = count_markers(all_disc_texts, NEGATIVE_MARKERS)

    ctrl_neg_total = sum(ctrl_neg_counts.values())
    disc_neg_total = sum(disc_neg_counts.values())
    ctrl_neg_rate = ctrl_neg_total / max(ctrl_total_words, 1) * 1000
    disc_neg_rate = disc_neg_total / max(disc_total_words, 1) * 1000

    print(f"\nControl:    {ctrl_neg_total} negative markers in {ctrl_total_words} words "
          f"({ctrl_neg_rate:.2f} per 1000 words)")
    print(f"Disclosure: {disc_neg_total} negative markers in {disc_total_words} words "
          f"({disc_neg_rate:.2f} per 1000 words)")
    print(f"Rate ratio: {disc_neg_rate / max(ctrl_neg_rate, 0.001):.3f}x")

    # per word comparison
    print(f"\n{'Word':<16} {'Control':>9} {'Disclosure':>12} {'Diff':>7} {'Ratio':>7}")
    print("-" * 55)
    all_neg_words = set(ctrl_neg_counts.keys()) | set(disc_neg_counts.keys())
    word_diffs = []
    for word in sorted(all_neg_words, key=lambda w: disc_neg_counts.get(w, 0) - ctrl_neg_counts.get(w, 0), reverse=True):
        c = ctrl_neg_counts.get(word, 0)
        d = disc_neg_counts.get(word, 0)
        ratio = d / max(c, 0.5)
        word_diffs.append((word, c, d, d - c, ratio))
        if c + d >= 3:  # only showing words with sufficient occurrences
            print(f"  {word:<14} {c:>9} {d:>12} {d-c:>+7} {ratio:>7.2f}x")

    # positive marker comparison 
    print("\n" + "=" * 60)
    print("2. POSITIVE-TONE MARKER FREQUENCY COMPARISON")
    print("=" * 60)

    ctrl_pos_counts, _ = count_markers(all_ctrl_texts, POSITIVE_MARKERS)
    disc_pos_counts, _ = count_markers(all_disc_texts, POSITIVE_MARKERS)

    ctrl_pos_total = sum(ctrl_pos_counts.values())
    disc_pos_total = sum(disc_pos_counts.values())
    ctrl_pos_rate = ctrl_pos_total / max(ctrl_total_words, 1) * 1000
    disc_pos_rate = disc_pos_total / max(disc_total_words, 1) * 1000

    print(f"\nControl:    {ctrl_pos_total} positive markers in {ctrl_total_words} words "
          f"({ctrl_pos_rate:.2f} per 1000 words)")
    print(f"Disclosure: {disc_pos_total} positive markers in {disc_total_words} words "
          f"({disc_pos_rate:.2f} per 1000 words)")
    print(f"Rate ratio: {disc_pos_rate / max(ctrl_pos_rate, 0.001):.3f}x")

    #  per-sample paired analysis 
    print("\n" + "=" * 60)
    print("3. PER-SAMPLE PAIRED MARKER ANALYSIS")
    print("=" * 60)

    sample_neg_diffs = []  # per-sample: disclosure neg rate (control neg rate)
    sample_pos_diffs = []
    penalized_neg_diffs = []  # for samples where disclosure got lower score
    non_penalized_neg_diffs = []

    for sid, data in pairs.items():
        ctrl_counts, ctrl_words = count_markers(data["control"], NEGATIVE_MARKERS)
        disc_counts, disc_words = count_markers(data["disclosure"], NEGATIVE_MARKERS)
        ctrl_rate = sum(ctrl_counts.values()) / max(ctrl_words, 1) * 1000
        disc_rate = sum(disc_counts.values()) / max(disc_words, 1) * 1000
        sample_neg_diffs.append(disc_rate - ctrl_rate)

        ctrl_pos, _ = count_markers(data["control"], POSITIVE_MARKERS)
        disc_pos, _ = count_markers(data["disclosure"], POSITIVE_MARKERS)
        ctrl_pos_rate = sum(ctrl_pos.values()) / max(ctrl_words, 1) * 1000
        disc_pos_rate = sum(disc_pos.values()) / max(disc_words, 1) * 1000
        sample_pos_diffs.append(disc_pos_rate - ctrl_pos_rate)

        # check if this sample was penalized
        ctrl_mean = np.mean(data["control_scores"]) if data["control_scores"] else 0
        disc_mean = np.mean(data["disclosure_scores"]) if data["disclosure_scores"] else 0
        if disc_mean < ctrl_mean:
            penalized_neg_diffs.append(disc_rate - ctrl_rate)
        else:
            non_penalized_neg_diffs.append(disc_rate - ctrl_rate)

    neg_diff_arr = np.array(sample_neg_diffs)
    pos_diff_arr = np.array(sample_pos_diffs)

    t_neg, p_neg = stats.ttest_1samp(neg_diff_arr, 0)
    t_pos, p_pos = stats.ttest_1samp(pos_diff_arr, 0)

    print(f"\nNegative marker rate difference (disclosure - control):")
    print(f"  Mean: {np.mean(neg_diff_arr):+.3f} per 1000 words")
    print(f"  t-statistic: {t_neg:.3f}, p-value: {p_neg:.4f}")
    sig_neg = "SIGNIFICANT" if p_neg < 0.05 else "not significant"
    print(f"  Result: {sig_neg}")

    print(f"\nPositive marker rate difference (disclosure - control):")
    print(f"  Mean: {np.mean(pos_diff_arr):+.3f} per 1000 words")
    print(f"  t-statistic: {t_pos:.3f}, p-value: {p_pos:.4f}")
    sig_pos = "SIGNIFICANT" if p_pos < 0.05 else "not significant"
    print(f"  Result: {sig_pos}")

    # comparing penalized vs non-penalized samples
    if penalized_neg_diffs and non_penalized_neg_diffs:
        pen_arr = np.array(penalized_neg_diffs)
        non_pen_arr = np.array(non_penalized_neg_diffs)
        t_comp, p_comp = stats.ttest_ind(pen_arr, non_pen_arr)
        print(f"\nPenalized samples (n={len(penalized_neg_diffs)}) negative marker diff: "
              f"{np.mean(pen_arr):+.3f}")
        print(f"Non-penalized samples (n={len(non_penalized_neg_diffs)}) negative marker diff: "
              f"{np.mean(non_pen_arr):+.3f}")
        print(f"Difference: t={t_comp:.3f}, p={p_comp:.4f}")

    # most differential words 
    print("\n" + "=" * 60)
    print("4. MOST DIFFERENTIAL WORDS (DATA-DRIVEN)")
    print("=" * 60)

    ctrl_word_counts, ctrl_total = compute_word_frequencies(all_ctrl_texts)
    disc_word_counts, disc_total = compute_word_frequencies(all_disc_texts)

    # compute per-word rate difference
    all_words = set(ctrl_word_counts.keys()) | set(disc_word_counts.keys())
    word_rate_diffs = []
    for word in all_words:
        c = ctrl_word_counts.get(word, 0)
        d = disc_word_counts.get(word, 0)
        if c + d < 5:
            continue
        ctrl_rate = c / ctrl_total * 10000
        disc_rate = d / disc_total * 10000
        word_rate_diffs.append((word, ctrl_rate, disc_rate, disc_rate - ctrl_rate, c, d))

    # sort by absolute difference
    word_rate_diffs.sort(key=lambda x: x[3], reverse=True)

    print("\nTop 20 words MORE frequent in disclosure reasoning:")
    print(f"{'Word':<18} {'Ctrl rate':>10} {'Disc rate':>10} {'Diff':>8} {'Count(C/D)':>12}")
    print("-" * 62)
    for word, cr, dr, diff, c, d in word_rate_diffs[:20]:
        print(f"  {word:<16} {cr:>10.2f} {dr:>10.2f} {diff:>+8.2f} {c:>5}/{d:<5}")

    print("\nTop 20 words LESS frequent in disclosure reasoning:")
    for word, cr, dr, diff, c, d in word_rate_diffs[-20:]:
        print(f"  {word:<16} {cr:>10.2f} {dr:>10.2f} {diff:>+8.2f} {c:>5}/{d:<5}")

    #  reasoning length comparison
    print("\n" + "=" * 60)
    print("5. REASONING LENGTH COMPARISON")
    print("=" * 60)

    ctrl_lengths = [len(t.split()) for t in all_ctrl_texts if t]
    disc_lengths = [len(t.split()) for t in all_disc_texts if t]
    t_len, p_len = stats.ttest_ind(ctrl_lengths, disc_lengths)
    print(f"Control reasoning length: {np.mean(ctrl_lengths):.1f} words (std={np.std(ctrl_lengths):.1f})")
    print(f"Disclosure reasoning length: {np.mean(disc_lengths):.1f} words (std={np.std(disc_lengths):.1f})")
    print(f"Difference: {np.mean(disc_lengths) - np.mean(ctrl_lengths):+.1f} words")
    print(f"t-statistic: {t_len:.3f}, p-value: {p_len:.4f}")

    print("\n" + "=" * 60)
    print("6. QUALITATIVE EXAMPLES: PENALIZED SAMPLES")
    print("=" * 60)
    print("(Samples where disclosure caused the largest score drop)")

    # most-penalized samples
    score_diffs = []
    for sid, data in pairs.items():
        ctrl_mean = np.mean(data["control_scores"]) if data["control_scores"] else None
        disc_mean = np.mean(data["disclosure_scores"]) if data["disclosure_scores"] else None
        if ctrl_mean is not None and disc_mean is not None:
            score_diffs.append((sid, disc_mean - ctrl_mean, ctrl_mean, disc_mean, data))

    score_diffs.sort(key=lambda x: x[1])

    for sid, diff, ctrl_s, disc_s, data in score_diffs[:5]:
        print(f"\n--- Sample {sid} (GT={data['ground_truth_score']}) ---")
        print(f"Control score: {ctrl_s:.2f}, Disclosure score: {disc_s:.2f} "
              f"(penalty: {diff:+.2f})")


        ctrl_text = data["control"][0] if data["control"] else ""
        disc_text = data["disclosure"][0] if data["disclosure"] else ""
        ctrl_words = set(tokenize_simple(ctrl_text))
        disc_words = set(tokenize_simple(disc_text))
        unique_to_disc = disc_words - ctrl_words
        negative_unique = [w for w in unique_to_disc if w in NEGATIVE_MARKERS]

        print(f"Negative words unique to disclosure reasoning: {negative_unique}")
        print(f"Control reasoning (first 200 chars): {ctrl_text[:200]}...")
        print(f"Disclosure reasoning (first 200 chars): {disc_text[:200]}...")

    # visuals
    print("\n" + "=" * 60)
    print("7. GENERATING VISUALIZATIONS")
    print("=" * 60)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Plot 1: negative marker rate comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Marker rates
    ax = axes[0]
    categories = ["Negative\nMarkers", "Positive\nMarkers"]
    ctrl_rates = [ctrl_neg_rate, ctrl_pos_rate]
    disc_rates = [disc_neg_rate, disc_pos_rate]
    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, ctrl_rates, width, label="Control", color="#3498db", alpha=0.8)
    ax.bar(x + width/2, disc_rates, width, label="Disclosure", color="#e74c3c", alpha=0.8)
    ax.set_ylabel("Rate per 1000 words", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_title("(A) Marker Word Rates", fontsize=11, fontweight="bold")
    ax.legend()

    # Panel B: Per-sample negative marker diff distribution
    ax = axes[1]
    ax.hist(neg_diff_arr, bins=25, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axvline(x=0, color="black", linewidth=1, linestyle="--")
    ax.axvline(x=np.mean(neg_diff_arr), color="#e74c3c", linewidth=2,
               label=f"Mean = {np.mean(neg_diff_arr):+.2f}")
    ax.set_xlabel("Negative Marker Rate Diff\n(Disclosure - Control, per 1000 words)", fontsize=10)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("(B) Per-Sample Negative Marker Diff", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)

    # Panel C: Top differential words
    ax = axes[2]
    top_disc_words = [(w, d) for w, cr, dr, d, c, dd in word_rate_diffs[:10]
                      if d > 0][:8]
    if top_disc_words:
        words, diffs_plot = zip(*top_disc_words)
        y_pos = range(len(words))
        ax.barh(y_pos, diffs_plot, color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words)
        ax.set_xlabel("Rate Difference (per 10000 words)", fontsize=10)
        ax.set_title("(C) Words More Frequent in\nDisclosure Reasoning", fontsize=11, fontweight="bold")
        ax.invert_yaxis()

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "reasoning_linguistic_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {plot_path}")

    # Plot 2: Penalized vs non-penalized comparison
    if penalized_neg_diffs and non_penalized_neg_diffs:
        fig, ax = plt.subplots(figsize=(8, 5))
        data_box = [penalized_neg_diffs, non_penalized_neg_diffs]
        bp = ax.boxplot(data_box, labels=[
            f"Penalized\n(n={len(penalized_neg_diffs)})",
            f"Not Penalized\n(n={len(non_penalized_neg_diffs)})"
        ], patch_artist=True)
        bp["boxes"][0].set_facecolor("#e74c3c")
        bp["boxes"][0].set_alpha(0.5)
        bp["boxes"][1].set_facecolor("#3498db")
        bp["boxes"][1].set_alpha(0.5)
        ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("Negative Marker Rate Diff\n(per 1000 words)", fontsize=11)
        ax.set_title("Negative Language Increase:\nPenalized vs Non-Penalized Samples",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plot_path2 = os.path.join(PLOTS_DIR, "reasoning_penalized_vs_not.png")
        plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {plot_path2}")

    analysis_output = {
        "n_paired_samples": len(pairs),
        "negative_markers": {
            "control_rate_per_1000": ctrl_neg_rate,
            "disclosure_rate_per_1000": disc_neg_rate,
            "rate_ratio": disc_neg_rate / max(ctrl_neg_rate, 0.001),
            "paired_t_stat": float(t_neg),
            "paired_p_value": float(p_neg),
        },
        "positive_markers": {
            "control_rate_per_1000": ctrl_pos_rate,
            "disclosure_rate_per_1000": disc_pos_rate,
            "rate_ratio": disc_pos_rate / max(ctrl_pos_rate, 0.001),
            "paired_t_stat": float(t_pos),
            "paired_p_value": float(p_pos),
        },
        "reasoning_length": {
            "control_mean_words": float(np.mean(ctrl_lengths)),
            "disclosure_mean_words": float(np.mean(disc_lengths)),
            "t_stat": float(t_len),
            "p_value": float(p_len),
        },
        "top_differential_words_disclosure": [
            {"word": w, "ctrl_rate": cr, "disc_rate": dr, "diff": d}
            for w, cr, dr, d, c, dd in word_rate_diffs[:30]
        ],
        "top_differential_words_control": [
            {"word": w, "ctrl_rate": cr, "disc_rate": dr, "diff": d}
            for w, cr, dr, d, c, dd in word_rate_diffs[-30:]
        ],
    }

    output_path = os.path.join(RESULTS_DIR, "reasoning_analysis.json")
    with open(output_path, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\nSaved analysis results to {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    run_reasoning_analysis()
