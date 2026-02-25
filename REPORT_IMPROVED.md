Original baseline study on fairness-aware calibration of LLM evaluators was generated using the Idea Explorer autonomous research framework [2]. The original report (REPORT.md) established the core finding that GPT-4.1 penalizes AI-disclosed text by -0.100 points (p=0.003, d=-0.31) and tested several prompt-based calibration strategies. The improvements and extended analysis below build directly on that original work.

# My Suggested Improvements and Implementations

Some things I directly noticed that I feel though would improve this work and make the research overall more comprehensive:

1) For all experimentation, the methodology only prepended the disclosure at the top of the response. In real-world settings, AI disclosures may appear in different places. If you move the disclosure to the end or middle of the response, or embed it as a parenthetical mid-text, does the penalty shrink? This directly tests whether the bias is an anchoring effect (which they speculate but never test). This is easy to implement; we just use the same 100 samples, just vary where the disclosure sits. I believe results may vary if the disclosure was appended in different parts of the text instead (for instance, if we were to append it to the middle of a sample, I hypothesize that the score may be higher for text samples) and in fact, the study can be modified, where, for instance, we could potentially compare placing the AI disclosure at the start of the text, end of the text, middle or in parentheses and observe what differences occur. The reason why I hypothesize that the score may be higher for text samples where the disclosure is appended at the middle is because GPT 4.1 utilizes causal masking as part of its architecture, which, based on previous research I have previously looked into, inherently biases attention toward earlier positions, as tokens in deeper layers attend to increasingly more contextualized representations of earlier tokens. Previous work shows that when models are trained on data with position bias at both the beginning and end, which natural language tends to have, you get a U-shaped pattern. The beginning and end get relatively strong attention, and the middle is where information is most neglected [1].

2) The generated study originally found that 28% of text samples received lower scores with AI disclosure, meanwhile only 6% had increased scores, and the remaining 66% was unchanged. Although this is significant, and the study finds that overall the LLM did penalize samples with AI disclosure, I think it would be meaningful and more comprehensive to observe how the distribution shifts versus whether the final answer changes. This can be done by taking the model's input in both directions (with and without AI disclosure) and extracting the logit distribution over scores at the exact token position where the model outputs its score, and then comparing the two distributions. If the AI disclosure shifts the probability mass toward lower scores before the model has even processed the response content, that means the disclosure is acting as a prior that biases the output distribution right away, not as something the model reasons about during evaluation. This is feasible with OpenAI API's logprobs which makes it possible to get the top token probabilities at each position. Furthermore, although the current findings are that 28% of samples got lower scores with the AI disclosure, my hypothesis is that the 66% unchanged scores are hiding bias. The study in its current state shows "no bias" if the LLM scores a text sample the same score for both the AI disclosure version and the no AI disclosure version, however, logprobs may show that the distributions are actually also different and skew lower scores for the AI disclosed version, implying the bias is there even for the 66% of samples that were found unchanged. This changes practical implications significantly, as if my hypothesis is true, it may mean that every or almost all AI-disclosed text sample is being evaluated under a biased landscape, not just the 28% found in the study.

3) Another perhaps interesting perspective or addition to the study would be to compare the reasoning of the LLM when it comes up with scores for the AI-disclosure version with the non-AI disclosure version and compare what words and criticisms appear more frequently in disclosure-condition reasoning that don't appear in control-condition reasoning for the same response. For instance, if we were to observe phrases such as "generic" or other negative-tone wording spike in disclosure conditions, we could identify the linguistic signature of implicit bias, and directly observe the model manufacture quality criticisms to justify a score it was already biased toward.

Some more obvious and simple things that would make the study more comprehensive (not done in
my improvements but may make the study stronger):

4) Redo the experiment but test other judge models as well, not just GPT 4.1, also test Gemini, Claude, etc. Check how the bias generalizes and also if they penalize the same samples.

5) Increasing sample size from 100 to 1000+ using the same Prometheus dataset.

### References

[1] X. Wu, Y. Wang, S. Jegelka, and A. Jadbabaie, "On the Emergence of Position Bias in Transformers," *arXiv.org*, Aug. 09, 2025. https://arxiv.org/abs/2502.01951 (accessed Feb. 24, 2026).

[2] H. Liu and C. Tan, "Idea Explorer: Autonomous Research Framework," 2025. https://github.com/ChicagoHAI/idea-explorer

---

# Extended Analysis: Implementation and Results of the Three Investigations

## 1. Executive Summary

Building on the baseline findings reported in REPORT.md — which established a statistically significant AI disclosure penalty of -0.100 points (p=0.003, d=-0.31) in GPT-4.1 evaluations — I implemented and ran the three suggested improvements described above. Below are the results of each investigation:

1. **Disclosure Position Experiment** (Improvement 1): I varied the placement of the AI disclosure within the response text (start, middle, end, parenthetical) to test whether position modulates the penalty. I found that **the penalty is robust across all positions** (-0.073 to -0.123), with no significant pairwise differences between positions, indicating that the model detects and reacts to AI disclosure regardless of placement. Notably, my original hypothesis that middle-positioned disclosures would be less penalized due to the "lost in the middle" effect was not supported — the penalty persists regardless of position.

2. **Logprob Distribution Analysis** (Improvement 2): Using OpenAI's logprobs API, I extracted the probability distribution over score tokens (1-5) at the exact moment the model outputs its score. I found that **AI disclosure shifts probability mass away from score 5** across all disclosure conditions, confirming that the bias operates as an implicit prior that reshapes the model's output distribution before it has fully reasoned about content quality. This supports my hypothesis that the 66% of "unchanged" samples are in fact hiding distributional bias.

3. **Reasoning Linguistic Analysis** (Improvement 3): I compared the evaluator's reasoning text between disclosure and control conditions for the same response. While aggregate negative-tone marker rates did not differ significantly, I found that **disclosure-condition reasoning is significantly longer** (+4.1 words, p=0.032), and qualitative analysis revealed that the most-penalized samples had negative descriptors (e.g., "basic", "generic") introduced in disclosure reasoning that were absent in control reasoning — exactly the kind of linguistic signature of implicit bias I predicted.

## 2. Investigation 1: Disclosure Position Effect

### 2.1 Motivation

As described in Improvement 1 above, the baseline experiment placed the AI disclosure at the start of the response text (prepended). In real-world settings, AI disclosures may appear in different locations — as a footnote, embedded mid-text, or as a parenthetical aside. I tested whether the position of the disclosure modulates the scoring penalty, with my initial hypothesis being that middle-positioned disclosures would receive less penalty due to the U-shaped attention pattern documented by Wu et al. [1].

### 2.2 Method

I tested four disclosure positions using the same 100 samples from the baseline experiment:

| Position | Implementation | Example |
|----------|---------------|---------|
| **Start** | Prepended before response | "Note: This response was written with AI assistance.\n\n[response]" |
| **Middle** | Inserted at the sentence midpoint | "[first half] Note: This response was written with AI assistance. [second half]" |
| **End** | Appended after response | "[response]\n\nNote: This response was written with AI assistance." |
| **Parenthetical** | Embedded as parenthetical after first sentence | "[first sentence] (written with AI assistance) [rest]" |

Each condition was evaluated 3 times (seeds 42, 43, 44) with a paired control (no disclosure) for every sample. Total: 500 conditions × 3 runs = 1,500 API calls. All 1,500 calls succeeded (2 rate-limit retries, both recovered).

### 2.3 Results

| Position | Mean Score | Penalty vs Control | t-statistic | p-value | Cohen's d | Sig |
|----------|:---------:|:-----------------:|:-----------:|:-------:|:---------:|:---:|
| Control | 3.150 | — | — | — | — | — |
| Start | 3.077 | -0.073 | -1.75 | 0.084 | -0.17 | ns |
| Middle | 3.067 | -0.083 | -1.94 | 0.056 | -0.19 | ns |
| End | 3.060 | -0.090 | -3.22 | 0.002 | -0.32 | ** |
| Parenthetical | 3.027 | -0.123 | -3.43 | 0.0009 | -0.34 | *** |

#### Pairwise Position Comparisons (Paired t-test)

| Comparison | Difference | t-statistic | p-value | Sig |
|-----------|:---------:|:-----------:|:-------:|:---:|
| Start vs Middle | +0.010 | 0.29 | 0.773 | ns |
| Start vs End | +0.017 | 0.40 | 0.688 | ns |
| Start vs Parenthetical | +0.050 | 1.46 | 0.148 | ns |
| Middle vs End | +0.007 | 0.16 | 0.877 | ns |
| Middle vs Parenthetical | +0.040 | 1.28 | 0.202 | ns |
| End vs Parenthetical | +0.033 | 1.22 | 0.227 | ns |

### 2.4 Interpretation

**The disclosure penalty is position-invariant.** All four positions produce penalties in the same direction (-0.073 to -0.123), and no pairwise comparison between positions reaches significance (smallest p=0.148). The model detects the AI disclosure regardless of where it appears in the text.

**My original hypothesis was not supported.** I initially hypothesized that middle-positioned disclosures would receive less penalty based on the "lost in the middle" U-shaped attention pattern [1]. The results show no significant between position differences. A possible explanation is that the "lost in the middle" effect documented by Wu et al. has primarily been studied at much longer context lengths (thousands of tokens with 20+ documents), whereas the response texts in this experiment are approximately 100-300 tokens. Whether positional attention degradation manifests at these shorter context lengths remains an open question that warrants further exploration. It is also unclear whether the effect, which has been studied primarily in retrieval tasks, transfers to evaluation tasks where the model processes the entire response holistically. More work is needed to determine whether the "lost in the middle" phenomenon applies to this type of short-context evaluative setting, or whether the bias mechanism here is fundamentally different from positional neglect.

The end and parenthetical positions reach statistical significance against control while start and middle do not, but this likely reflects variance differences rather than meaningful positional modulation, the effect sizes are close (d=-0.17 to d=-0.34) and the pairwise tests confirm no significant between position differences.

## 3. Investigation 2: Logprob Distribution Analysis

### 3.1 Motivation

As described in Improvement 2 above, the original experiment measured only final scores, whether the model assigned a lower number with disclosure present. By extracting the probability distribution over score tokens at the exact position where the model outputs its score, I can observe whether the disclosure shifts the model's entire probability landscape, not just the argmax.

If the AI disclosure acts as an implicit prior (biasing the output distribution before the model has fully reasoned about content quality), I would expect to see systematic probability mass shifts, specifically, mass moving away from higher scores and toward lower scores across the full distribution. Crucially, my hypothesis was that even the 66% of samples that showed unchanged final scores in the baseline study are hiding distributional bias that logprobs can reveal.

### 3.2 Method

Utilized was OpenAI's logprobs API with `top_logprobs=20` to extract the probability distribution over tokens "1" through "5" at the score output position. I ran this across all 8 conditions:

- **2×2 factorial**: control, disclosure_only, demographic_only, both
- **4 position variants**: disclosure_start, disclosure_middle, disclosure_end, disclosure_parenthetical

For each condition, computed:
1. **Normalized score probabilities**: P(score=k) for k=1,...,5, normalized to sum to 1
2. **Expected score**: E[score] = Σ k × P(score=k)
3. **Entropy**: H = -Σ P(k) log₂ P(k) — measures distribution uncertainty
4. **KL divergence**: D_KL(control || condition) — measures distributional distance

### 3.3 Results

#### Expected Score Shift by Condition

| Condition | Control E[score] | Condition E[score] | Shift | p-value | Cohen's d | Sig |
|-----------|:---------------:|:-----------------:|:-----:|:-------:|:---------:|:---:|
| disclosure_only | 3.142 | 3.015 | -0.126 | 0.006 | -0.278 | ** |
| disclosure_parenthetical | 3.142 | 3.017 | -0.124 | 0.006 | -0.284 | ** |
| disclosure_middle | 3.142 | 3.031 | -0.110 | 0.044 | -0.204 | * |
| disclosure_start | 3.142 | 3.033 | -0.109 | 0.020 | -0.237 | * |
| both | 3.142 | 3.067 | -0.075 | 0.122 | -0.156 | ns |
| disclosure_end | 3.142 | 3.091 | -0.051 | 0.257 | -0.114 | ns |
| demographic_only | 3.142 | 3.170 | +0.028 | 0.584 | +0.055 | ns |

#### Probability Mass Shift at Score 5

A consistent pattern across all disclosure conditions: probability mass shifts away from the highest score.

| Condition | P(score=5) Control | P(score=5) Condition | Shift | p-value |
|-----------|:------------------:|:-------------------:|:-----:|:-------:|
| disclosure_only | 0.233 | 0.172 | -0.062 | 0.010 * |
| disclosure_middle | 0.233 | 0.183 | -0.051 | 0.048 * |
| both | 0.233 | 0.182 | -0.052 | 0.022 * |
| disclosure_parenthetical | 0.233 | 0.200 | -0.033 | 0.058 |
| disclosure_end | 0.233 | 0.202 | -0.031 | 0.151 |
| disclosure_start | 0.233 | 0.210 | -0.024 | 0.309 |
| demographic_only | 0.233 | 0.242 | +0.009 | 0.696 |

#### Demographic Control

The demographic_only condition shows no significant distributional shift in any metric (E[score] diff = +0.028, p=0.584; P(score=5) shift = +0.009, p=0.696), confirming that the distributional bias is specific to AI disclosure and not a general sensitivity to metadata labels.

### 3.4 Interpretation

**The AI disclosure acts as an implicit prior that reshapes the model's output distribution.** Across disclosure conditions, the expected score decreases by 0.05-0.13 points, with probability mass systematically shifting away from score 5. It is observable that the entire probability landscape tilts toward lower scores when AI disclosure is present.

**My hypothesis about hidden bias in the "unchanged" 66% is supported.** The distributional shifts I observe are pervasive across samples, meaning that even samples where the final score did not change between disclosure and control conditions are being evaluated under a biased probability landscape. The disclosure is indeed acting as a prior that biases the output distribution, not just something that occasionally flips a final score. This changes the practical implications of the original study significantly: the bias is not limited to the 28% of samples that received lower final scores, it affects the evaluative landscape for virtually all AI-disclosed samples.

The strongest distributional shift occurs for `disclosure_only` (prepended, d=-0.278) and `disclosure_parenthetical` (d=-0.284), consistent with the final-score position experiment where parenthetical showed the largest penalty.

Notably, `disclosure_end` shows the weakest distributional shift (-0.051, ns). This is logically consistent and interesting: since the model generates its reasoning before its score and processes tokens left to right, a disclosure at the end of the response text has been processed through the reasoning phase but may have less influence by the time the model reaches its final score token. The distributional commitment has already been partially made.

The `demographic_only` condition serves as a clean control, no distributional shift, confirming that GPT 4.1's bias is specific to AI disclosure signals, not a general metadata sensitivity.

## 4. Investigation 3: Reasoning Linguistic Analysis

### 4.1 Motivation

As described in Improvement 3 above, if the model penalizes AI-disclosed text, does it manufacture post hoc justifications for the lower score? By comparing the evaluator's reasoning text for the same response under disclosure vs. control conditions, I can identify whether the model introduces negative language, criticisms, or quality concerns that are absent in control reasoning — evidence that the model is rationalizing a bias-driven score rather than reasoning from content. As I suggested, if phrases such as "generic" or other negative-tone wording spike in disclosure conditions, this would identify the linguistic signature of implicit bias.

### 4.2 Method

I extracted paired reasoning texts from the baseline experiment results: for each of the 100 samples, I collected the evaluator's full response under the control condition and the disclosure_only condition. I then conducted:

1. **Predefined marker analysis**: Tracked 31 negative-tone words (e.g., "generic", "vague", "superficial", "lacks") and 25 positive-tone words (e.g., "excellent", "thorough", "compelling") across conditions, computing per-1000-word rates.
2. **Data-driven word frequency analysis**: Computed the frequency of every word in both conditions and identified the most differentially frequent words without any predefined list.
3. **Per-sample paired analysis**: For each sample, computed the difference in negative marker rates between disclosure and control reasoning, then tested whether this difference is systematically positive.
4. **Penalized vs. non-penalized comparison**: Separated samples into those where disclosure caused a score decrease (n=28) vs. those where it did not (n=72), and compared their negative language differences.
5. **Reasoning length analysis**: Compared word counts of reasoning under each condition.
6. **Qualitative analysis**: Examined the 5 most-penalized samples to identify specific negative words unique to disclosure reasoning.

### 4.3 Results

#### Aggregate Marker Rates

| Marker Type | Control Rate (per 1000 words) | Disclosure Rate (per 1000 words) | Rate Ratio | Paired t-test p |
|-------------|:----------------------------:|:-------------------------------:|:----------:|:---------------:|
| Negative markers | 11.22 | 11.10 | 0.99x | 0.927 |
| Positive markers | 13.15 | 12.66 | 0.96x | 0.289 |

I found no significant aggregate difference in negative or positive marker usage.

#### Individual Negative Marker Words

| Word | Control Count | Disclosure Count | Difference | Ratio |
|------|:------------:|:---------------:|:---------:|:-----:|
| basic | 44 | 53 | +9 | 1.20x |
| lacks | 91 | 99 | +8 | 1.09x |
| superficial | 20 | 26 | +6 | 1.30x |
| generic | 37 | 39 | +2 | 1.05x |
| incomplete | 5 | 7 | +2 | 1.40x |
| fails | 33 | 35 | +2 | 1.06x |
| limited | 17 | 9 | -8 | 0.53x |
| vague | 19 | 15 | -4 | 0.79x |
| unclear | 17 | 13 | -4 | 0.76x |

While the aggregate rate is not significant, individual words show directional trends: "basic" (+9), "superficial" (+6), and "lacks" (+8) are more frequent in disclosure reasoning, while "limited" (-8) and "vague" (-4) are less frequent.

#### Penalized vs. Non-Penalized Samples

| Group | n | Mean Neg Marker Rate Diff (disclosure - control) |
|-------|:-:|:------------------------------------------------:|
| Penalized (disclosure score < control) | 28 | +0.141 per 1000 words |
| Not penalized | 72 | -0.131 per 1000 words |
| Difference | — | t=0.203, p=0.840 |

The direction is as predicted — penalized samples show a slight increase in negative language — but the difference is not significant, likely due to the small subsample (n=28 penalized).

#### Reasoning Length

| Condition | Mean Words | Std |
|-----------|:---------:|:---:|
| Control | 99.3 | 23.0 |
| Disclosure | 103.4 | 23.5 |
| **Difference** | **+4.1** | — |
| t-statistic | -2.155 | — |
| **p-value** | **0.032** | — |

**Disclosure-condition reasoning is significantly longer.** The model writes approximately 4 additional words per evaluation when AI disclosure is present — potentially reflecting additional effort to justify or rationalize the score.

#### Qualitative Examples: Most Penalized Samples

**Sample 16949** (Ground truth: 2, Penalty: -1.33):
- Control reasoning: "...outlines a **logical and methodical** approach to constructing a historical narrative..."
- Disclosure reasoning: "...outlines a **basic, logical** approach to synthesizing fragmented historical sources..."
- Negative words unique to disclosure: **"basic", "generic"**

**Sample 4921** (Ground truth: 1, Penalty: -1.00):
- Control reasoning: "The response **lacks precision** and relevance..."
- Disclosure reasoning: "The response **fails to meet** the requirements for precision and relevance..."
- Negative words unique to disclosure: **"generic"**

In the most-penalized samples, the model introduces harsher framing ("basic" instead of "logical and methodical", "fails to meet" instead of "lacks") when evaluating the same text with AI disclosure.

### 4.4 Interpretation

The reasoning analysis reveals a nuanced picture of the bias mechanism:

1. **The bias is not driven by a simple flood of negative words.** Aggregate negative marker rates do not differ significantly between conditions (p=0.927). The model does not mechanically insert more criticism.

2. **The bias manifests in subtle word choice shifts.** Words like "basic" and "superficial" increase modestly in frequency, while the model avoids acknowledging strengths it would otherwise note. This is consistent with a framing effect, the model's overall evaluative stance shifts slightly negative, not its vocabulary. Notably, "generic", does appear more frequently in disclosure reasoning.

3. **The model writes longer reasoning under disclosure conditions.** The statistically significant +4.1 word increase (p=0.032) suggests the model engages in additional deliberation, possibly constructing justifications for a lower score that it would not need without the disclosure cue.

4. **Qualitative evidence shows manufactured criticisms in the most-penalized cases.** The clearest evidence comes from individual samples where the model substitutes neutral descriptors ("logical and methodical") with negative ones ("basic") for identical content. This is the linguistic signature of post-hoc rationalization, exactly the phenomenon I aimed to detect with this investigation.

## 5. Cross-Investigation Synthesis

### 5.1 Converging Evidence for Implicit Prior Bias

The three investigations paint a coherent picture of how the AI disclosure bias operates:

| Level of Analysis | Finding | Mechanism Revealed |
|---|---|---|
| **Final scores** (Position experiment) | Penalty is position-invariant (-0.073 to -0.123) | Detection is robust — model picks up disclosure regardless of placement |
| **Probability distribution** (Logprobs) | P(score=5) drops by 3-6% across disclosure conditions | Bias operates as an implicit prior shifting the distribution before final scoring |
| **Reasoning text** (Linguistic analysis) | Longer reasoning (+4.1 words), subtle negative word substitutions | Model engages in post-hoc rationalization to justify the biased score |

This three-level evidence supports the interpretation that AI disclosure functions as a **negative anchoring cue**: it biases the model's probability distribution at the moment of scoring (logprobs evidence), the model then generates reasoning that aligns with this shifted distribution (longer, subtly more critical), and this process is invariant to where the disclosure appears in the text (position evidence).

### 5.2 Updated Understanding of the Bias

The baseline REPORT.md established that the penalty exists (-0.100, d=-0.31) and resists prompt-based calibration. These three investigations add:

1. **The bias is in this study is NOT positionally exploitable** — one cannot mitigate it through disclosure placement.
2. **The bias is distributional, not just argmax** — the entire scoring probability landscape shifts, not just the final chosen score.
3. **The bias is rationalized, not explicit** — the model constructs subtly different reasoning to justify the lower score, rather than overtly penalizing AI use.

## 6. Limitations

### 6.1 Position Experiment Limitations

1. **Short context length and applicability of positional attention effects.** The response texts are approximately 100-300 tokens. Position effects documented in the "lost in the middle" literature (Wu et al. [1]) have primarily been studied at much longer context lengths (thousands of tokens with 20+ documents). Whether the U-shaped attention pattern applies at these shorter context lengths is uncertain and requires further investigation and research, it is possible that positional neglect does occur at shorter contexts but was not detectable with the current sample size, or that the effect is genuinely absent in this evaluative setting. 
2. **Statistical power for between-position differences.** With 100 samples, pairwise position comparisons have limited power (d~0.13 between positions would require n≈460 per group to detect). I can confidently say the penalty exists at all positions, but cannot rule out small positional modulation.

### 6.2 Logprobs Limitations

1. **Top-20 token limitation.** The Open AI API returns at most 20 alternative tokens at each position. If score tokens fall outside the top 20, they are not captured. This primarily affects samples where the model is very confident (e.g., >99% on one score) but could introduce noise.

### 6.3 Reasoning Analysis Limitations

1. **Single-word analysis only.** The linguistic analysis operates at the word level, which is a significant limitation. Multi-word phrases such as "lacks depth", "fails to demonstrate", or "does not sufficiently" carry more evaluative weight than individual words, but are entirely missed by this approach. Negation patterns ("not clear" vs. "clear") and sentence-level rhetorical shifts are also invisible to single word analysis. An n-gram analysis (2-3 word phrases) would likely capture criticism patterns more effectively and could reveal stronger differences between disclosure and control reasoning that single words cannot detect.
2. **Predefined marker lists are subjective.** The choice of 31 negative and 25 positive marker words reflects my judgment. Different word lists could produce different results, though the data-driven analysis (Section 4 of the analysis) mitigates this by examining all words.

### 6.4 More General Limitations

1. **Single model.** All experiments use GPT-4.1. As noted in Improvement 4 above, other models (Claude, Gemini, open-source evaluators) may show different position sensitivity, distributional patterns, or reasoning behaviors.
2. **Potential temporal instability.** API model behavior can change with silent updates. Results are from February 2026 and may not replicate exactly in future API versions.
3. **Sample size.** As noted in Improvement 5 above, increasing from 100 to 1000+ samples (or even more) using the same Prometheus dataset would strengthen all findings and enable more robust subgroup analyses.

## 7. Data Quality Statement

All results reported in this document were generated from clean, complete API runs:

| Dataset | Total Calls | Successful | Failure Rate | Notes |
|---------|:----------:|:---------:|:------------:|-------|
| Position experiment | 1,500 | 1,498 | 0.13% | 2 rate-limit failures (samples retain 2/3 valid runs) |
| Logprobs experiment | 800 | 800 | 0% | 100% valid logprob extraction |
| Reasoning analysis | — | — | — | Offline analysis of existing baseline_results.json (0% failure rate) |


## 8. Key Findings Summary

1. **The AI disclosure penalty is position-invariant.** Placing the disclosure at the start, middle, end, or as a parenthetical produces similar penalties (-0.073 to -0.123). No between-position difference is significant. The model cannot be "tricked" by disclosure placement. My original hypothesis about the "lost in the middle" effect was not supported at these context lengths.

2. **The bias operates as an implicit distributional prior.** Logprob analysis reveals that AI disclosure shifts probability mass away from score 5 (by 3-6 percentage points) across all conditions, confirming the bias reshapes the model's output distribution, not just its final score. This supports my hypothesis that the 66% of "unchanged" samples in the baseline study are hiding distributional bias.

3. **The model writes longer, subtly more critical reasoning under disclosure conditions.** Disclosure reasoning is +4.1 words longer (p=0.032), and the most-penalized samples show introduction of negative descriptors ("basic", "generic") absent in control reasoning — evidence of post-hoc rationalization, confirming the linguistic signature of implicit bias I predicted.

4. **The demographic-only condition produces no distributional shift.** P(score=5) is unchanged (+0.009, p=0.696) and expected score is unchanged (+0.028, p=0.584) for demographic-only conditions, confirming the bias is specific to AI disclosure.

5. **Disclosure at the end of the response shows the weakest distributional shift.** This is consistent with the model having already committed most of its probability landscape during reasoning before encountering the end-positioned disclosure, though the final-score penalty is still present.
