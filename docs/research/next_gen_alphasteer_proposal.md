# Next-Generation AlphaSteer: LocalAlphaSteer
## From Global Linear Null-Space to Locally Nonlinear Constraints

---

## Executive Summary

AlphaSteer achieves single-round jailbreak defense by projecting the steering direction onto the global null-space of benign activations. This proposal upgrades the global linear null-space constraint to a **family of K locally-gated projectors**, yielding a locally nonlinear constraint that adapts to the current hidden state while preserving AlphaSteer's training-derived, plug-and-play character. We name this method **LocalAlphaSteer**.

The key principle: **K=1 recovers AlphaSteer exactly.** Every added component is motivated by one concrete failure mode of the global projector.

---

## Stage 0: Problem Re-Statement and Design Space

### T0.1 — Most Precise Problem Definition

> Find a **training-derived single-space activation steering method** that inherits AlphaSteer's plug-and-play, constraint-internalised design, but replaces the global linear null-space projector with **K locally-gated null-space projectors** whose soft-attention weights depend on the current hidden state, so that (i) benign anchor points are kept point-invariant and their neighbourhoods are kept approximately invariant, (ii) malicious hidden states are still steered toward refusal, and (iii) the effective steering matrix is a smooth nonlinear function of the hidden state.

### T0.2 — Design Space

```
Single-space methods
├── Fixed-direction type  (Curveball style)                    ← PRIMARY
│   ├── A1. LocalAlphaSteer — soft mixture of K local matrices    ← RECOMMENDED
│   └── A2. Kernel-regression steering matrix                    ← FALLBACK A
└── Direction-field type  (ODESteer style)
    └── B1. Learned vector-field with local benign barrier        ← FALLBACK B

Dual-space methods (backup only)
└── C1. Separate benign chart + malicious separation spaces
```

### T0.3 — Priority Order

| Rank | Direction | Reason |
|------|-----------|--------|
| 1 | A1 LocalAlphaSteer | AlphaSteer drop-in, K=1 exact recovery, minimal extra complexity |
| 2 | A2 Kernel-regression | More flexible, heavier inference cost |
| 3 | B1 ODESteer-style | Elegant but requires ODE solver at inference |
| 4 | C1 Dual-space | Only if single-space is systematically falsified |

**Not pursued first**: direct non-linear flow methods, explicit semantic invariance loss, or heavy dual-space architectures.

---

## Stage 1: Focused Literature Survey

### T1.1 — Training-derived single-space representation tools

| Tool | Core idea | Single-space | Local constraint | Trainable | AlphaSteer link |
|------|-----------|--------------|-----------------|-----------|-----------------|
| Contractive AE | Jacobian-norm regulariser | ✓ | ✓ (Jacobian) | ✓ | P replaced by learned manifold tangent |
| Denoising AE | Local manifold via noise | ✓ | ✓ | ✓ | Encodes benign chart implicitly |
| Graph Laplacian repr. | Smooth repr. on kNN graph | ✓ | ✓ | ✓ | Cluster-consistent projectors |
| Local LDA | Discriminative local bases | ✓ | ✓ | ✓ | Local null-space extension of AlphaSteer |
| K-means + local PCA | Cluster then project | ✓ | ✓ | via clustering | Direct implementation of LocalAlphaSteer |
| Nyström approx. | Kernel PCA, scalable | ✓ | ✓ | ✓ | Smooth interpolation of projectors |

**Verdict**: K-means + local null-space PCA is the most natural single-space tool for LocalAlphaSteer because it directly generalises AlphaSteer's SVD-based null-space construction, and the soft-attention gating provides the smooth interpolation.

### T1.2 — Non-linear steering tools in the same space

| Tool | Style | Position-dependent | Trainable | Notes |
|------|-------|--------------------|-----------|-------|
| Curveball | Fixed direction (approx.) | No | No | Single global matrix |
| LocalAlphaSteer | Curveball-style | Yes (soft weights) | Yes | Proposed here |
| ODESteer | Direction field | Yes | Yes | Heavier inference |
| Position-dependent LQR | Linear quadratic regulator | Yes | Yes | Needs cost function |
| Potential gradient | Learned scalar field | Yes | Yes | Hard to train |

**Verdict**: LocalAlphaSteer sits in the Curveball column but lifts it to be position-dependent via soft gating. This is the minimum extension needed.

### T1.3 — Local constraint internalisation tools

| Tool | Mechanism | Notes |
|------|-----------|-------|
| Local projector family (proposed) | Cluster centroids + per-cluster SVD null-space | Direct extension of AlphaSteer |
| Mixture-of-experts controller | Gated linear controllers | Close to LocalAlphaSteer |
| Kernel regression operator | Nadaraya-Watson weighting | Smooth but expensive |
| Graph zero-update field | Laplacian smoothing | Requires explicit graph |
| Barrier / repulsion field | Penalty near benign anchors | Requires tuning |

### T1.4 — Adaptability summary

All single-space tools scored against the key criteria:

| Tool | Single-space | Local point-invariant | AlphaSteer narrative | Curveball-style | Cost |
|------|:---:|:---:|:---:|:---:|:---:|
| K-means + local null-space | ✓ | ✓ | ✓ (direct upgrade) | ✓ | Low |
| Kernel regression | ✓ | ✓ | ~ | ✓ | Medium |
| ODESteer / vector field | ✓ | ~ | ✗ | ✗ | High |
| Dual-space | ✗ | ✓ | ~ | ✓ | High |

---

## Stage 2: Candidate Scheme Construction

### Candidate A1 — LocalAlphaSteer (Recommended)

**Type**: Curveball-style, single space, soft mixture of K local steering matrices.

#### Mathematics

**Offline** (training-time), for each layer `ℓ`:

1. Cluster benign embeddings into K groups using K-means:
   $$\{c_k, \mathcal{C}_k\}_{k=1}^K = \text{KMeans}(H_b^{(\ell)},\, K)$$

2. For each cluster, compute the local null-space projector:
   $$P_k = Q_k Q_k^\top, \quad Q_k = \text{NullSpace}\!\left(H_b^{(\ell)}[\mathcal{C}_k]\right)$$

3. Solve for a shared (or per-cluster) delta matrix via regularised pseudoinverse:
   $$\tilde\Delta = \arg\min_\Delta \|H_h^{(\ell)} \bar P \Delta - r^{(\ell)}\|^2 + \lambda\|\Delta\|^2, \quad \bar P = \tfrac{1}{K}\sum_k P_k$$

4. Compute K local steering matrices:
   $$M_k = P_k \tilde\Delta, \quad k = 1, \ldots, K$$

5. Save `centroids` $(K \times D)$ and `local_steering_matrices` $(K \times D \times D)$ per layer.

**Online** (inference-time), at each steered layer:

$$h \;\in\; \mathbb{R}^D$$

$$w_k(h) = \frac{\exp\!\left(\operatorname{sim}(h,\, c_k)/\tau\right)}{\sum_{j=1}^K \exp\!\left(\operatorname{sim}(h,\, c_j)/\tau\right)}$$

$$M(h) = \sum_{k=1}^K w_k(h)\, M_k$$

$$h' = h + \lambda\,(h \cdot M(h))$$

where $\operatorname{sim}$ is cosine similarity and $\tau$ is the temperature.

**Special case K=1**: $w_1 = 1$, $M(h) = M_1 = P_1\tilde\Delta = P\tilde\Delta$ — identical to AlphaSteer.

#### Local point-invariance

For a benign anchor $b \in \mathcal{C}_k$, $b \approx c_k$, so $w_k(b) \approx 1$ and $M(b) \approx M_k = P_k\tilde\Delta$. Since $P_k$ is the null-space projector of the benign cluster containing $b$:
$$b \cdot M_k = b \cdot P_k\tilde\Delta = (b P_k)\tilde\Delta \approx 0$$
because $b$ lies (approximately) in the row-space of $H_b[\mathcal{C}_k]$, which is orthogonal to the null-space of $H_b[\mathcal{C}_k]$.

For off-cluster benign states $b'$ (e.g., $b' \approx c_j$, $j \neq k$), $w_j(b') \approx 1$, and $b' P_j \approx 0$ by the same argument. The update decays smoothly to zero across the entire benign manifold.

#### Relationship to AlphaSteer

LocalAlphaSteer is a strict generalisation: the global projector $\hat P$ is replaced by a soft mixture $P(h)=\sum_k w_k(h)P_k$ that is locally adapted to the current activation. The constraint narrative is identical ("project $\tilde\Delta$ onto benign null-space"), but the null-space is now local.

---

### Candidate A2 — NadarayaSteer (Kernel-regression steering)

**Type**: Curveball-style, single space, Nadaraya-Watson weighted steering matrix.

#### Mathematics

$$M(h) = \frac{\sum_{i} k(h, h_i^b)\, M_i}{\sum_{i} k(h, h_i^b)}, \quad k(h, h') = \exp\!\left(-\|h - h'\|^2 / 2\sigma^2\right)$$

where $M_i = P(h_i^b)\tilde\Delta$ are per-sample local steering matrices.

Computationally expensive at inference. Reduces to LocalAlphaSteer when centroids are used as representatives.

**Relationship to AlphaSteer**: more flexible but heavier; not the primary choice.

---

### Candidate B1 — FlowSteer (ODESteer-style direction field)

**Type**: Direction-field, single space, learned neural ODE on the hidden state.

#### Mathematics

$$\dot h = f_\theta(h) = v_\theta(h) \cdot \mathbf{1}[\text{malicious}(h)] \cdot (I - P_{\text{benign}}(h))$$

Steering is a continuous trajectory $h(t)$ integrated from $h(0)=h$ over $t \in [0,1]$.

Advantages: can express multi-step P5 dynamics naturally. Disadvantages: requires ODE solver at inference, harder to plug-in.

**Relationship to AlphaSteer**: loses the elegant one-shot steering narrative.

---

## Stage 3: Comparison and Final Recommendation

### Evaluation matrix

| Criterion | A1 LocalAlphaSteer | A2 NadarayaSteer | B1 FlowSteer |
|-----------|:-----------------:|:-----------------:|:------------:|
| Single-space elegance | ✓✓ | ✓✓ | ✓✓ |
| AlphaSteer narrative continuity | ✓✓ | ✓ | ✗ |
| Local point-invariance | ✓✓ | ✓✓ | ✓ |
| Non-linear steering | ✓ (via soft weights) | ✓ | ✓✓ |
| Training complexity | Low | Medium | High |
| Inference complexity | O(K·D²) | O(N·D²) | O(steps·D) |
| P1/P3/P5 verifiability | ✓✓ (K ablation) | ✓ | ✓ (P5) |
| K=1 recovery | Exact | ~ | ✗ |

### **Recommendation: A1 LocalAlphaSteer**

> If implementation starts today, implement A1 LocalAlphaSteer with K=4.

**Why not A2**: Nadaraya-Watson per-sample weighting is O(N·D²) at inference and does not offer a clean K=1 recovery. LocalAlphaSteer already captures the key benefit (position-dependent projector) with a tractable K.

**Why not B1**: ODESteer-style methods break the AlphaSteer plug-in narrative and require an ODE solver. They are the right tool if P5 (multi-step dynamics necessity) becomes a hard requirement, but we should test single-shot first.

---

## Stage 4: Mathematical Draft of LocalAlphaSteer

### T4.1 — Representation space

We work directly in the residual stream activation space $h \in \mathbb{R}^D$. No learned encoder is added. The "space" is the activation space of each steered transformer layer. This preserves the AlphaSteer plug-in property.

The K-means clustering and per-cluster SVD define a piece-wise linear atlas of the benign activation manifold, approximating the local chart structure required by P2.

### T4.2 — Local benign constraint

**Benign anchor set**: $\mathcal{B}^{(\ell)} = \{h_i^b\}$ — last-token hidden states of benign prompts at layer $\ell$.

**Clustering**: $\{c_k, \mathcal{C}_k\}_{k=1}^K = \text{KMeans}(\mathcal{B}^{(\ell)}, K)$.

**Per-cluster null-space projector**:
$$P_k = Q_k Q_k^\top, \quad Q_k = \text{right singular vectors of } H_b[\mathcal{C}_k] \text{ below threshold}$$

**Local composite projector** (inference-time):
$$P(h) = \sum_{k=1}^K w_k(h)\, P_k, \quad w_k(h) = \mathrm{softmax}_k\!\left(\frac{h \cdot c_k}{\|h\|\|c_k\|\tau}\right)$$

**Local point-invariance loss** (used to verify, not trained):
$$\mathcal{L}_{\text{local}} = \frac{1}{|\mathcal{B}^{(\ell)}|} \sum_{h_i^b \in \mathcal{B}^{(\ell)}} \|h_i^b \cdot P(h_i^b) \cdot \tilde\Delta\|^2$$

This loss is minimised by construction through the null-space property.

### T4.3 — Steering definition

**Offline computation** (stored per layer):

$$M_k = P_k\,\tilde\Delta^{(\ell)}, \quad k = 1,\ldots,K \tag{local steering matrix}$$

**Inference-time update**:

$$h' = h + \lambda \left(h \cdot \sum_{k=1}^K w_k(h)\, M_k\right)$$

Equivalently, this is a single matrix-vector product with the state-dependent matrix $M(h) = \sum_k w_k(h) M_k$:

$$h' = h + \lambda\, (h \cdot M(h))$$

### T4.4 — Training objectives (offline)

The "training" here is the closed-form offline computation (no gradient descent required, consistent with AlphaSteer):

1. **Benign preservation**: $P_k$ is the null-space of $H_b[\mathcal{C}_k]$ → $H_b[\mathcal{C}_k] \cdot P_k \approx 0$ (exact by construction).

2. **Safety enhancement**: $\tilde\Delta$ is computed by regularised pseudoinverse:
   $$\tilde\Delta = \arg\min_\Delta \|H_h \bar P \Delta - r\|^2 + \lambda_{\text{reg}} \|\bar P \Delta\|_F^2$$
   where $\bar P = \frac{1}{K}\sum_k P_k$ is the average projector used for Δ estimation.

3. **Cluster balance** (soft constraint): K-means is run with balanced init (K-means++) to ensure clusters cover the benign manifold uniformly.

**Training pipeline**:
1. Extract benign activations $H_b^{(\ell)}$ and harmful activations $H_h^{(\ell)}$ at each steered layer.
2. Run K-means on $H_b^{(\ell)}$ to get centroids $\{c_k\}$ and cluster assignments $\{\mathcal{C}_k\}$.
3. For each cluster $k$, compute $P_k$ via SVD of the cluster benign matrix.
4. Compute $\bar P$ and solve for $\tilde\Delta$ via regularised pseudoinverse.
5. Compute $M_k = P_k\tilde\Delta$ for each $k$.
6. Save $\{c_k\}$ and $\{M_k\}$ for each layer.

**Inference pipeline**:
1. Load model with LocalAlphaSteer decoder layers.
2. At each steered layer, compute soft weights $w_k(h)$ and apply $h' = h + \lambda (h \cdot M(h))$.
3. No gradient computation needed.

---

## Stage 5: Experimental Plan

### T5.1 — Alignment with existing experiments

| Existing result | How it supports LocalAlphaSteer |
|----------------|--------------------------------|
| P2 strongly supported (local projector family is real) | Direct motivation for K>1 clustering |
| P4 local distortion supported | Shows global P̂ introduces local error LocalAlphaSteer corrects |
| G1/G2 benign utility scores | Can be directly reused with K ablation |
| 1.5A hard jailbreak baseline | Adversarial test for LocalAlphaSteer with K=4 |

### T5.2 — Minimum viable experiments

| Experiment | Question answered | Metric |
|-----------|------------------|--------|
| E1: K ablation (K=1,2,4,8) | Does K>1 improve local benign preservation? | Benign local violation rate |
| E2: LocalAlphaSteer vs AlphaSteer (same λ) | Does local projector improve trade-off? | ASR↓ + Alpaca-Eval↑ |
| E3: P(h) vs P̂ cosine distance | Is P(h) actually different from global P̂ for hard cases? | Direction compatibility μ(h) |
| E4: Benign anchor update magnitude | Does ||h' - h|| ≈ 0 for benign h? | Mean benign update norm |

### T5.3 — Key diagnostic metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| μ(h) | cos(steering_direction, benign_null_space) | μ(h_benign) → 1, μ(h_malicious) < 0.5 |
| Benign local violation | Mean ||h_benign · M(h)|| | < 0.01 × ||h_benign|| |
| ASR@K | Attack success rate at K projectors | Monotone ↓ with K (or plateaus) |
| Alpaca-Eval@K | Benign utility at K projectors | Stays ≥ K=1 baseline |
| Hard-case malicious push | Mean ||v_malicious|| | > threshold |

### T5.4 — Success / failure criteria

**Supporting the primary line (LocalAlphaSteer works)**:
- E1 shows K>1 reduces benign local violation while maintaining ASR reduction.
- E3 shows P(h) deviates from P̂ for borderline/jailbreak inputs.
- E2 shows Alpaca-Eval improves or stays flat while ASR decreases relative to K=1.

**Single-space beginning to fail**:
- Benign local violation cannot be reduced below a threshold even with K=8.
- P(h) tracks malicious directions for hard jailbreaks (direction compatibility μ > 0.8 for malicious).
- Utility at K>1 is systematically worse than at K=1.

---

## Stage 6: Single-Space Failure Criteria and Dual-Space Fallback

### T6.1 — When to abandon single-space

The single-space approach should be abandoned if **all three** of the following hold:

1. **Geometric incompatibility**: $\mu(h)$ for benign inputs remains > 0.3 even at K=8 (the safety direction is not orthogonal to any benign local subspace).
2. **Irreducible benign violation**: Benign local violation > 1% of ||h|| for a significant fraction of benign inputs, regardless of K.
3. **No utility-safety improvement**: The Alpaca-Eval / ASR trade-off curve for K>1 is always dominated by the K=1 (AlphaSteer) baseline.

### T6.2 — Minimal dual-space fallback

If single-space fails, the minimal dual-space extension is:

**Step 1**: Train a lightweight encoder $\phi: \mathbb{R}^D \to \mathbb{R}^d$ ($d \ll D$) that separates benign from malicious via contrastive loss.

**Step 2**: Compute local projectors in the $z=\phi(h)$ space.

**Step 3**: Map the projected direction back to the original space via the pseudo-inverse of the Jacobian $J_\phi(h)$:
$$\Delta h = J_\phi(h)^\dagger \cdot \Delta z$$

The final steering still acts in the original activation space, preserving the AlphaSteer forward-pass interface.

---

## Implementation Roadmap

| Week | Milestone |
|------|-----------|
| W1 | Implement `local_steering_utils.py` (K-means + per-cluster null-space); validate K=1 == AlphaSteer |
| W1 | Implement `LocalAlphaLlama/Gemma/Qwen` model classes with soft-gated steering |
| W2 | Run E4 (benign anchor update magnitude) and E1 (K ablation) on llama3.1 |
| W2 | Run E2 (LocalAlphaSteer vs AlphaSteer) on standard benchmarks |
| W3 | Run E3 (direction compatibility μ) to probe P1 and P3 |
| W3 | Analyse results; decide whether to scale K or pivot |
| W4 | If single-space holds: write up and submit; if fails: begin dual-space fallback |
