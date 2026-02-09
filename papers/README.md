# Initial Survey on Reasoning and Efficiency: Recent Advances in Deployable Language Models

This repository curates and critically reviews recent work on efficient and robust inference for reasoning-capable language models, with a focus on quantization, test-time adaptation, and scaling strategies. Link to my Zotero library for this project: [Zotero](https://www.zotero.org/groups/6415084/llm-efficiency-research/library).

## Summary Table

| Paper | Focus | Key Contribution | Limitations / Open Questions |
|-------|-------|------------------|------------------------------|
| Deng et al. (2508.02180) | Quantized Test-Time Adaptation | ZOA: zeroth-order optimization for quantized model adaptation under domain shift | Assumes detectable domain shift; limited to ViT/ResNet architectures; no reasoning tasks evaluated |
| Liu et al. (2504.04823) | Quantization of reasoning models | First systematic study showing reasoning models tolerate aggressive quantization (W4A16 lossless); reveals task difficulty × bit-width interaction; exposes model-origin sensitivity** | Heavy math bias (AIME/GSM8K); no fine-grained error localization in CoT; static quantization only; ignores safety/calibration impacts |
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
- **Problem**: ...
- **Approach**: ...
- **Insight**: ...
- **Critique**: ...
- **Link**: [arXiv](https://arxiv.org/abs/2510.02611)

### 4. Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models 

### Problem
> **The tension**: Reasoning models generate 10–100× longer CoTs than standard LLMs → massive inference cost → quantization essential for deployment. **But**: theoretical concern that quantization errors compound across long reasoning chains, catastrophically degrading correctness. *No prior work empirically validated this for modern reasoning models.*

### Approach
- **Scope**: 7 models (1.5B–70B) including DeepSeek-R1 variants, QwQ-32B, Qwen3-8B
- **Quantization**: 12 algorithms across weight-only (AWQ/GPTQ), KV cache (QuaRot), weight-activation (SmoothQuant/FlatQuant) at 3–8 bits
- **Benchmarks**: AIME-120 (hard math), MATH-500, GSM8K (easy math), GPQA-Diamond (science), LiveCodeBench (coding)
- **Key design**: Controlled ablation on model origin (distillation vs. RL training from same base), CoT length analysis, scaling studies

### Core Insights (Takeaways)
| Insight | Why it matters for *your* research |
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
- **Problem**: ...
- **Approach**: ...
- **Insight**: ...
- **Critique**: ...
- **Link**: [arXiv](https://arxiv.org/abs/2510.16805)


...

## Emerging Themes & Research Directions
- Theme 1: Quantization disproportionately harms reasoning (vs. standard tasks)
- Theme 2: Test-time diversity (via temperature or traces) unlocks latent capability
- Theme 3: Efficiency ≠ just speed; must preserve reasoning fidelity













