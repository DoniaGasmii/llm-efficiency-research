# Initial Survey on Reasoning and Efficiency: Recent Advances in Deployable Language Models

This repository curates and critically reviews recent work on efficient and robust inference for reasoning-capable language models, with a focus on quantization, test-time adaptation, and scaling strategies. Link to my Zotero library for this project: [Zotero](https://www.zotero.org/groups/6415084/llm-efficiency-research/library).

## Summary Table

| Paper | Focus | Key Contribution | Limitations / Open Questions |
|------|-------|------------------|------------------------------|
| Deng et al. (2508.02180) | Quantized TTA | ZOA: zeroth-order adaptation for quantized models | Assumes domain shift detection; limited architecture scope |
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

### Direction 1: SHORTER
- **RL methods**: O1-Pruner (length+accuracy rewards), DAST (dynamic token budgets)
- **SFT methods**: TokenSkip (semantic skipping), TALE (optimal token budget search)
- **Latent reasoning**: Implicit-KD (hidden-state distillation), Coconut (continuous feedback)
- ✅ Best for: Max compression (10-100× token reduction)
- ⚠️ Tradeoff: Interpretability loss; harder debugging

### Direction 2: SMALLER
- **Distillation**: Mix (long+short CoT data), DLCoT (segmented simplification)
- **Quantization**: 8-bit ≈ lossless; 4-bit degrades complex reasoning
- **Pruning**: ❌ Generally harmful to reasoning (unlike standard LLMs)
- **RL from scratch**: Open-RS (1.5B matches o1-preview via GRPO)
- ✅ Best for: Deployment on edge devices
- ⚠️ Tradeoff: Requires high-quality teacher data or massive RL compute

### Direction 3: FASTER
- **Efficient sampling**: ϕ-Decoding (future path simulation), Fast Best-of-N (early rejection)
- **Self-consistency**: RPC (perplexity-guided convergence), Path-Consistency (prefix reuse)
- **Decomposition**: AoT (DAG-based subproblems), AR (atomic reasoning trees)
- ✅ Best for: Real-time applications
- ⚠️ Tradeoff: Added overhead from controllers/value models may offset gains

### Critical Insights
- Small models CAN reason well (Qwen2.5 series) — capacity ≠ bottleneck
- Pruning fails for reasoning → suggests distributed representation reliance
- Safety-efficiency tension: Shorter CoTs may skip safety self-correction (H-CoT)

### Future Directions
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
- **Problem**: ...
- **Approach**: ...
- **Insight**: ...
- **Critique**: ...
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
