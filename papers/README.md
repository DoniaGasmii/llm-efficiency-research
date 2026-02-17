# Initial Survey on Reasoning and Efficiency: Recent Advances in Deployable Language Models

This repository curates and critically reviews recent work on efficient and robust inference for reasoning-capable language models, with a focus on quantization, test-time adaptation, and scaling strategies. Link to my Zotero library for this project: [Zotero](https://www.zotero.org/groups/6415084/llm-efficiency-research/library).

## Summary Table 

| Paper | Focus | Key Contribution | Limitations / Open Questions |
|-------|-------|------------------|------------------------------|
| Deng et al. (2508.02180) | Quantized Test-Time Adaptation | ZOA: zeroth-order optimization for quantized model adaptation under domain shift | Assumes detectable domain shift; limited to ViT/ResNet architectures; no reasoning tasks evaluated |
| Liu et al. (2504.04823) | Quantization of reasoning models | First systematic study showing reasoning models tolerate aggressive quantization (W4A16 lossless); reveals task difficulty × bit-width interaction; exposes model-origin sensitivity | Heavy math bias (AIME/GSM8K); no fine-grained error localization in CoT; static quantization only; ignores safety/calibration impacts |
| Rakka et al. (2510.16805) | Mixed-precision quantization survey for LMs | Formal taxonomy distinguishing true MP (varying bitwidths *within* layers) from misnomers; systematic framework comparison showing 4-bit avg precision preserves accuracy (<2% perplexity Δ) while 2-bit fails catastrophically (>55% Δ); exposes LM-specific constraints (search-based optimization infeasible at scale → reliance on sensitivity heuristics) | No standardized benchmark suite (heterogeneous eval protocols); hardware support lagging (dequantization overhead on GPUs); KV-cache quantization largely unexplored despite dominating long-context memory; missing energy-accuracy trade-off curves |
| ... | ... | ... | ... |

## Paper Reviews

### 1. Test-Time Model Adaptation for Quantized Neural Networks 
- **Problem**: ...
- **Approach**: ...
- **Insight**: ...
- **Critique**: ...
- **Link**: [arXiv](https://doi.org/10.48550/arXiv.2508.02180)

### 2. Efficient Reasoning Models: A Survey (Feng et al., 2025)

**Problem**: Reasoning models generate excessively long CoTs → computational/memory overhead across 3 dimensions:
- Length redundancy (verbose chains)
- Size redundancy (massive params)
- Latency redundancy (slow decoding)

**Core Framework**: 3 orthogonal efficiency directions:
1. **SHORTER** ; compress CoT length
2. **SMALLER** ; compact models with strong reasoning
3. **FASTER** ; accelerate decoding

#### Direction 1: SHORTER
- **RL methods**: O1-Pruner (length+accuracy rewards), DAST (dynamic token budgets)
- **SFT methods**: TokenSkip (semantic skipping), TALE (optimal token budget search)
- **Latent reasoning**: Implicit-KD (hidden-state distillation), Coconut (continuous feedback)
- ✅ Best for: Max compression (10-100× token reduction)
- ⚠️ Tradeoff: Interpretability loss; harder debugging

#### Direction 2: SMALLER
- **Distillation**: Mix (long+short CoT data), DLCoT (segmented simplification)
- **Quantization**: 8-bit ≈ lossless; 4-bit degrades complex reasoning
- **Pruning**: ❌ Generally harmful to reasoning (unlike standard LLMs)
- **RL from scratch**: Open-RS (1.5B matches o1-preview via GRPO)
- ✅ Best for: Deployment on edge devices
- ⚠️ Tradeoff: Requires high-quality teacher data or massive RL compute

#### Direction 3: FASTER
- **Efficient sampling**: ϕ-Decoding (future path simulation), Fast Best-of-N (early rejection)
- **Self-consistency**: RPC (perplexity-guided convergence), Path-Consistency (prefix reuse)
- **Decomposition**: AoT (DAG-based subproblems), AR (atomic reasoning trees)
- ✅ Best for: Real-time applications
- ⚠️ Tradeoff: Added overhead from controllers/value models may offset gains

#### Critical Insights
- Small models CAN reason well (Qwen2.5 series) — capacity ≠ bottleneck
- Pruning fails for reasoning → suggests distributed representation reliance
- Safety-efficiency tension: Shorter CoTs may skip safety self-correction (H-CoT)

#### Future Directions
- Safety-aware compression (avoid bypassing guardrails)
- Memory-efficient architectures (KV cache compression)
- Training efficiency via tiny high-quality datasets (LIMO: 817 samples)
- Sustainability metrics (carbon footprint reduction)

**Link**: https://arxiv.org/abs/2504.10903
**Repo**: https://github.com/fscdc/Awesome-Efficient-Reasoning-Models

### 3. On the Role of Temperature Sampling in Test-Time Scaling 

> Sampling at multiple temperatures unlocks hard problems that no single temperature can solve, giving base LLMs a free +7.3 points over standard TTS — and closing the gap with RL-trained models.

---

**Problem:** Test-Time Scaling (TTS) assumes that generating more samples K always improves reasoning. Prior work treated temperature as a minor hyperparameter and kept it fixed. This paper shows both assumptions are wrong: at large K, accuracy plateaus completely, and fixing temperature means you're only exploring a fraction of the model's actual reasoning boundary.

---

**Core Idea:** Different temperatures solve *different subsets* of hard problems. Temperature isn't just a creativity dial — it's an independent axis of reasoning diversity. A question that's unsolvable at T=0.7 with 1,024 samples may be trivially solvable at T=0.9. By splitting your sample budget across multiple temperatures instead of one, you enlarge the total set of questions the model can solve — without any extra training.

---

**Method:** Given a compute budget, divide samples evenly across temperatures T=0.0 to T=1.2 (in 0.1 steps, skipping 0.1–0.3 which add no new coverage). Run all temperatures in parallel, then use a verifier or voting to select the best answer. An optional early-exit variant (multi-temperature voting) identifies easy questions early and stops sampling them, cutting compute by ~26–54% at no accuracy cost.

---

**Results:**
- Averaged over Qwen3 (0.6B → 8B) and 5 benchmarks: **+7.3 points** over single-temperature TTS
- Qwen3-4B on AIME 2025: **+13.3 points** (60.0% → 73.3%)
- Scaling K from 1,024 → 13,312 at fixed T: **+0%** (hard plateau)
- Scaling T across 13 temperatures at 1,024 samples each: **+6.67%** (on the same question set)
- Base Qwen3-4B with temperature scaling **matches** RL-trained Polaris-4B on AIME 2025 Pass@All
- Efficient voting method cuts compute **31–54%** on MATH500, **~78%** on Hi-ToM, with negligible accuracy loss

---

**Why It Works:** The model's solvable question set has four natural categories: *easy* (any temperature works), *medium* (all temperatures work but rarely), *hard* (only specific temperatures work — this is the key tier), and *impossible* (no temperature can solve them, training is needed). The entropy analysis confirms this: for easy/medium questions, correct traces have noticeably lower entropy than incorrect ones — the model "knows it knows." For hard questions this signal breaks down entirely, which means uncertainty-guided decoding strategies (discard high-entropy traces) are only safe for easy problems. Temperature scaling works because it ensures the hard-but-solvable questions find their preferred temperature, rather than being abandoned because they never fired at the one temperature you chose.

---

**Critique:**

- 📄 *Paper admits:*
  - Compute cost scales linearly with number of temperatures — 12 temperatures = 12× the cost of single-T, which is only acceptable if you have a strong verifier to exit easy questions early
  - Voting-based early exit has a known failure mode: majority vote isn't always right, especially for hard problems where the model is rarely correct
  - Hi-ToM shows inconsistent scaling behavior — some questions get "solved" spuriously by weaker models, so the benchmark may not cleanly separate real reasoning gains from noise
  - Variable temperature *within* a single trace (as opposed to across traces) is unexplored and flagged as future work

- 🧠 *My read:*
  - The "+7.3 points" headline is averaged across model sizes including 0.6B, where gains are large but arguably less meaningful (small models have high variance). The more honest headline might be Qwen3-8B's average of +4.8 points
  - The RL comparison is compelling but narrow — only 30 AIME 2025 problems, only one RL model (Polaris-4B). Matching on 3 unsolved questions out of 30 could be noise
  - They use GPT-5 to verify AIME reasoning traces (filtering "lucky guesses") — this is rigorous and worth noting, but it also means results may not replicate in setups without a strong verifier
  - The paper treats temperatures 0.0–1.2 uniformly but gives no guidance on how to choose the subset for a new domain or model — the "skip 0.1–0.3" heuristic is derived empirically on Qwen3 and may not generalize
  - The core phenomenon (different temperatures = different solvable sets) is shown but not deeply explained. *Why* does T=0.9 unlock a specific AIME problem that T=0.7 can't? The entropy analysis describes the effect but doesn't explain the cause at the level of attention or reasoning steps

---

**Open Questions:**
- What determines a question's "preferred temperature"? Is it related to problem structure, required reasoning depth, or something else?
- Does temperature scaling compose with search-based TTS (Tree of Thoughts, MCTS)? Could you scale both the branching factor and temperature?
- Can a model learn to predict its own preferred temperature per question — making this adaptive rather than brute-force?
- The "impossible" category (questions no temperature can solve) — how large is it in practice, and is it fixed or model-size dependent?
- Does this transfer to multimodal or agentic settings, or is it specific to autoregressive text reasoning?

---

**Link:** [arXiv:2510.02611](https://arxiv.org/abs/2510.02611)

### 4. Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models 

### Problem
> **The tension**: Reasoning models generate 10–100× longer CoTs than standard LLMs → massive inference cost → quantization essential for deployment. **But**: theoretical concern that quantization errors compound across long reasoning chains, catastrophically degrading correctness. *No prior work empirically validated this for modern reasoning models.*

### Approach
- **Scope**: 7 models (1.5B–70B) including DeepSeek-R1 variants, QwQ, Qwen3
- **Quantization**: 12 algorithms across weight-only (AWQ/GPTQ), KV cache (QuaRot), weight-activation (SmoothQuant/FlatQuant) at 3–8 bits
- **Benchmarks**: AIME-120 (hard math), MATH-500, GSM8K (easy math), GPQA-Diamond (science), LiveCodeBench (coding)
- **Key design**: Controlled ablation on model origin (distillation vs. RL training from same base), CoT length analysis, scaling studies

### Core Insights (Takeaways)
| Insight | Why it matters for *our* research |
|---------|-----------------------------------|
| **W4A16 is effectively lossless** (≤1% drop) even on AIME | → Quantization *isn't* the bottleneck for reasoning quality; focus should shift to *where* errors occur in CoT rather than bit-width alone |
| **Hard tasks suffer 4× more degradation** at low bits (W4A4) | → Suggests *error accumulation scales with reasoning depth*; opportunity to model error propagation as function of CoT length/task complexity |
| **Model origin > architecture**: Distilled vs. RL-trained models from same base show divergent quantization tolerance | → Training dynamics imprint quantization robustness; implies *quantization-aware reasoning distillation* could be a novel training paradigm |
| **No "overthinking" artifact**: Quantized models don't generate longer CoTs when accuracy drops | → Rules out one failure hypothesis; errors likely stem from *premise corruption* early in CoT rather than compensatory verbosity |

### Critique (Gaps → Project Opportunities)
| Limitation | Research Direction It Reveals |
|------------|-------------------------------|
| **No error localization**: Don't know *where* in CoT quantization fails (premise vs. calculation vs. conclusion) | → Build *quantization error tracing*: instrument models to detect step-wise confidence collapse; could enable adaptive bit-width per reasoning phase |
| **Static quantization only**: All layers quantized uniformly | → **Dynamic quantization scheduler**: allocate higher bits to "critical" layers identified via attention entropy or gradient sensitivity during CoT |
| **Math-heavy benchmarks**: Ignores logical/causal/temporal reasoning | → Test quantization on *reasoning type spectrum*: does symbolic logic degrade differently than arithmetic under quantization? |
| **No safety analysis**: Quantized models may become *confidently wrong* | → Study *calibration shift*: do quantized reasoners maintain uncertainty awareness? Critical for deployment |

### Emerging Themes & Research Directions (Cross-Paper Synthesis)

Based on this paper + quantization/reasoning literature, these are **actionable project directions** with varying risk/reward profiles:

### High-Impact Directions (Novel + Feasible)

| Direction | Core Idea | Why Now? | First Experiment |
|-----------|-----------|----------|------------------|
| **Adaptive Quantization Scheduler** | Dynamically allocate bit-width *per reasoning step* based on uncertainty signals (attention entropy, token probability dispersion) | Static quantization wastes bits on "easy" steps; this paper shows error accumulation is non-uniform | Instrument DeepSeek-R1 to log confidence per CoT step → correlate with quantization error → train lightweight controller to modulate bits |
| **Quantization-Robust Reasoning Distillation** | Modify distillation objectives to explicitly preserve quantization tolerance (e.g., add quantization noise during teacher→student training) | This paper shows model origin matters → implies training dynamics can be engineered for robustness | Distill Qwen2.5 → student with/without quantization-aware loss → compare W4 performance on AIME |
| **Error Propagation Modeling** | Formalize quantization error as stochastic process over CoT length: *Δaccuracy = f(bit-width, CoT_length, task_complexity)* | Current work is empirical → a predictive model would guide deployment decisions (e.g., "W4 safe up to 200-token CoT on GSM8K") | Fit regression on Liu et al.'s data + extend to new tasks; validate on held-out reasoning benchmarks |

#### Riskier but Transformative

| Direction | Core Idea | Challenge |
|-----------|-----------|-----------|
| **Quantization as Reasoning Regularizer** | Hypothesis: mild quantization *improves* reasoning by suppressing overconfident hallucinations | Requires careful task selection; may only help on specific reasoning types |
| **Hardware-Aware Reasoning Architectures** | Co-design models where reasoning pathways align with quantization-friendly operations (e.g., avoid sensitive attention patterns) | Requires hardware co-design; longer timeline |

#### Quick Validation Experiments (1–2 weeks)
1. **Reproduce Liu et al.'s W4A16 result on a new task** (e.g., temporal reasoning from TemporalWiki) → does losslessness generalize beyond math?
2. **Error localization pilot**: Run quantized DeepSeek-R1 on 50 AIME problems → manually annotate *where* CoT first diverges from gold solution → compute distribution (premise/calculation/conclusion)
3. **Calibration check**: Compare ECE (Expected Calibration Error) of BF16 vs. W4 models on reasoning tasks → do quantized models become overconfident?

- **Link**: [arXiv](https://arxiv.org/abs/2504.04823)



### 5. Mixed-Precision Quantization for Language Models: Techniques and Prospects

#### Core Problem
- Exponential LM scaling → unsustainable compute/memory demands
- Uniform low-bit quantization degrades accuracy in sensitive components
- Need adaptive precision allocation balancing efficiency/accuracy

#### Key Contribution: Formal Taxonomy
- **MPW**: Mixed-precision weights only (W-int{4,8}/A-FP16)
- **MPW+UPA**: Mixed weights + uniform activations (W-int{4,8}/A-int8)
- **MPW+MPA**: Mixed weights + mixed activations (W-int{4,8}/A-{int8,BF16})
- *Excludes misnomers*: W-only quantization or uniform W/A at different precisions

#### Framework Landscape (Top Performers)
| Framework | Strategy | Avg Bits | Perplexity Δ (LLaMA2-13B) |
|-----------|----------|----------|---------------------------|
| MixLLM    | Salience-driven channels | 4-bit | +1.0-2.0% |
| SqLLM     | Hessian-guided + sparse outliers | 4-bit | +1.0-2.0% |
| CMPQ      | Outlier preservation | 4-bit | +13% (LLaMA3-8B) |
| BitMod    | Hardware-aware grouping | 4-bit | +2.7-4.5% |
| ResQ      | PCA subspace separation | 4-bit | +2.7-4.5% |

#### Critical Insight
- 4-bit avg precision → near-baseline accuracy achievable
- 2-bit avg precision → catastrophic degradation (>55% perplexity ↑)
- MPW/MPW+UPA frameworks outperform MPW+MPA → activations harder to quantize than weights
- LM-specific constraint: Search-based optimization (RL/NAS) infeasible at billion-param scale → reliance on sensitivity heuristics

#### Limitations / Open Questions
- No standardized benchmark suite for MXPLM comparison (heterogeneous eval protocols)
- Hardware support lagging: Most GPUs lack native mixed-precision execution → dequantization overhead
- KV-cache quantization largely unexplored despite dominating long-context memory
- Trade-off analysis missing: Energy savings vs. accuracy loss curves not systematically reported

#### Future Directions
1. Hardware-aware design (TPU/GPU support for fine-grained mixed precision)
2. Activation/KV-cache quantization robustness
3. Scalable global optimization (beyond layer-wise heuristics)
4. Dynamic precision scheduling for edge/cloud deployment

#### Link [arXiv](https://arxiv.org/abs/2510.16805)


...

## Emerging Themes & Research Directions
- Theme 1: Quantization disproportionately harms reasoning (vs. standard tasks)
- Theme 2: Test-time diversity (via temperature or traces) unlocks latent capability
- Theme 3: Efficiency ≠ just speed; must preserve reasoning fidelity













