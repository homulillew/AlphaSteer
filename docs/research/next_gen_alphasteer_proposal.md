# Next-Generation AlphaSteer: From Global Linear Null-Space to Local Nonlinear Constraints

> **A Single-Space Activation Steering Method with Trained Local Projectors**

---

## 1. Executive Summary

This document proposes **LocalAlphaSteer**, a next-generation activation steering method that upgrades AlphaSteer's global linear null-space constraint to a family of locally-gated nonlinear projectors, while preserving AlphaSteer's core elegance: constraints enter the steering body, the method is trained end-to-end, and inference remains plug-and-play.

**Core upgrade path:**

| Aspect | AlphaSteer | LocalAlphaSteer |
|--------|-----------|-----------------|
| Constraint space | Global linear null-space $\hat{P}$ | Local projector family $\{P_k\}_{k=1}^K$ with kernel gates |
| Steering form | $h' = h + \lambda (h \cdot \tilde{\Delta}\hat{P})$ | $h' = h + \lambda \sum_k w_k(h)(h \cdot \tilde{\Delta}_k P_k)$ |
| Benign invariance | Global: $\Delta H_b \approx 0$ | Local: $\Delta h \approx 0$ for $h \in \mathcal{N}_\epsilon(h_b)$ |
| Training | SVD-based closed-form | SVD initialization + end-to-end fine-tuning |
| Degeneracy | $K=1$ recovers AlphaSteer | By construction |

**Key design choice:** Curveball-style (fixed-direction-per-anchor) within a learned single representation space, with kernel-weighted local projectors providing position-dependent gating. This is preferred over ODESteer-style continuous dynamics for its simplicity, lower training cost, and natural alignment with AlphaSteer's narrative.

---

## 2. Precise Problem Statement

> **Find or construct a single trained representation space $\mathcal{Z} = \phi_\theta(\mathcal{H})$ in which (a) local benign constraints—point invariance and neighborhood approximate invariance—are expressible as a family of locally-gated null-space projectors, and (b) safety steering is realized as a projection of a refusal direction field onto the local constraint-compatible subspace; the method must degenerate to AlphaSteer when $K=1$ and the projectors collapse to a single global projector.**

This inherits from AlphaSteer:
- **Constraint internalization:** constraints are part of the steering operator, not post-hoc
- **Utility–safety decomposition:** steering ≈ 0 on benign states, steering → refusal on malicious states
- **Plug-and-play inference:** no model weight modification, only activation intervention

This upgrades AlphaSteer by addressing:
- **P2 (local subspace family):** benign constraints are not a single global null-space but a family of local subspaces
- **P4 (global projection compromise):** a global projector compromises between local distortion and over-conservatism
- **Locality:** point invariance + neighborhood approximate invariance, not semantic invariance

---

## 3. Phase 0: Design Space and Priority

### T0.1 Method Type Tree

```
Activation Steering Methods
├── Single-Space Methods (PRIORITY)
│   ├── Type A: Single-Space + Fixed Direction (Curveball-style)
│   │   ├── A1: Kernel-Weighted Local Projectors (★ RECOMMENDED)
│   │   ├── A2: Contrastive-AE + Local Null-Space
│   │   └── A3: Mixture-of-Experts Local Projectors
│   ├── Type B: Single-Space + Direction Field (ODESteer-style)
│   │   ├── B1: Learned Vector Field with Barrier Constraints
│   │   └── B2: Neural ODE with Lyapunov Benign Basin
│   └── Type C: Hybrid (fixed direction + local gate)
│       └── C1: Gated Local Projector with Fixed Refusal Direction
└── Dual-Space Methods (BACKUP ONLY)
    ├── D1: Separate Chart Space + Separation Space
    └── D2: Shared Encoder with Dual Heads
```

### T0.2 Priority Ranking

| Priority | Direction | Rationale |
|----------|-----------|-----------|
| 1 (FIRST) | A1: Kernel-Weighted Local Projectors | Most natural AlphaSteer upgrade; single space; trained; local constraints internalized |
| 2 | A2: Contrastive-AE + Local Null-Space | Elegant but heavier training; single space |
| 3 | C1: Gated Local Projector | Good fallback if A1's soft gating is insufficient |
| 4 | B1: Learned Vector Field | Powerful but complex; loses fixed-direction elegance |
| BACKUP | D1/D2: Dual-Space | Only if single-space is systematically falsified |

### T0.3 Deprioritized Directions

- **Direct dual-space from the start:** Contradicts single-space priority
- **Heavy ODESteer dynamics:** Overcomplicates inference; loses plug-and-play property
- **Explicit semantic invariance:** Too ambitious; local point invariance is the pragmatic target
- **Pure kernel methods without training:** Lack adaptability; cannot learn task-specific structure

---

## 4. Phase 1: Focused Literature and Tool Survey

### T1.1 Trained Single-Space Representation Learning Tools

| Tool | Core Idea | Single-Space? | Local Constraint? | Trainable? | AlphaSteer Relation | Recommendation |
|------|-----------|:---:|:---:|:---:|-----|:---:|
| **Contractive Autoencoder (CAE)** | Penalizes Jacobian norm → locally flat encoding | ✓ | ✓ (via Jacobian) | ✓ | Jacobian penalty = local null-space regularizer | ★★★ |
| **Denoising Autoencoder (DAE)** | Reconstructs from corrupted input → learns manifold | ✓ | Partial | ✓ | Implicit local smoothness | ★★ |
| **Local Tangent Space Alignment** | Aligns local tangent spaces across neighbors | ✓ | ✓ | Partial | Direct local subspace estimation | ★★ |
| **Graph Laplacian Regularization** | Penalizes variation across graph neighbors | ✓ | ✓ | ✓ | Smoothness ≈ local invariance | ★★★ |
| **Jacobian Regularization** | Directly penalizes $\|\partial f/\partial x\|$ at anchors | ✓ | ✓ | ✓ | Local invariance of steering at benign points | ★★★ |
| **Metric Learning (Triplet/Contrastive)** | Learns distances that separate classes | ✓ | Partial | ✓ | Can enforce benign cluster compactness | ★★ |
| **Learnable Kernel Methods (RFF/Nyström)** | Approximate kernel in learned feature space | ✓ | ✓ | ✓ | Kernel weights for local projectors | ★★★ |
| **Local Invertible Mapping** | Normalizing flow / invertible residual net | ✓ | ✓ | ✓ | Could provide exact local linearization | ★★ |

### T1.2 Nonlinear Steering in the Same Space

| Tool | Core Idea | Style | Local Constraint Integration | Training Cost | Recommendation |
|------|-----------|-------|------------------------------|:---:|:---:|
| **Kernel-weighted projector family** | $\sum_k w_k(h) P_k$ with soft RBF weights | Curveball | Natural: each $P_k$ is a local null-space | Low–Med | ★★★ |
| **Position-dependent linear controller** | $\Delta h = A(h) \cdot d_{\text{refusal}}$ | Curveball | Gate matrix $A(h)$ can enforce local zeroing | Low | ★★★ |
| **Neural ODE steering field** | $\dot{h} = f_\theta(h)$ | ODESteer | Constraints via Lyapunov / barrier | High | ★★ |
| **Potential function gradient** | $\Delta h = -\nabla V(h)$ | ODESteer | Benign points at potential minima | Med | ★★ |
| **Mixture-of-Experts (MoE) local matrices** | Router selects local steering matrix | Curveball | Each expert is a local projector | Med | ★★★ |

### T1.3 Local Constraint Internalization Tools

| Tool | Core Idea | Expressiveness | Integration Ease | Recommendation |
|------|-----------|:---:|:---:|:---:|
| **Local projector family $\{P_k\}$** | K projectors, kernel-weighted selection | High | Natural AlphaSteer upgrade | ★★★ |
| **Piecewise linear controller** | Different linear steering in Voronoi cells | Medium | Clean but discontinuous at boundaries | ★★ |
| **MoE-style local matrices** | Soft router over K steering matrices | High | Standard deep learning pattern | ★★★ |
| **Graph-based zero-update field** | Zero steering at benign graph nodes, propagate | Medium | Requires graph construction at inference | ★ |
| **Kernel regression local operators** | $P(h) = \sum_i K(h, h_i) P_i$ | High | Continuous; matches kernel-weighted projectors | ★★★ |
| **Anchor-invariant local controllers** | Hard constraint: $\Delta(h_{\text{anchor}}) = 0$ | Medium | Easy to enforce as loss term | ★★★ |

### T1.4 Adaptability Summary

The most promising tool combination for a single-space method is:

1. **Kernel-weighted local null-space projectors** — directly generalizes AlphaSteer's $\hat{P}$
2. **Jacobian regularization at benign anchors** — enforces local point invariance
3. **Graph Laplacian smoothness** — ensures neighborhood approximate invariance
4. **Learnable RBF/Nyström kernel** — provides soft gating that is trainable

---

## 5. Phase 2: Three Single-Space Candidate Methods

### Candidate A1: **LocalAlphaSteer** (Kernel-Weighted Local Projectors) — ★ RECOMMENDED

**Type:** Curveball-style (fixed-direction-per-anchor within each local projector)

**Core idea:** Replace AlphaSteer's single global projector $\hat{P}$ with $K$ local projectors $\{P_k\}_{k=1}^K$, softly gated by kernel weights $w_k(h)$. Each projector $P_k$ is the null-space projector of a local cluster of benign activations. The refusal direction is projected through the locally-weighted projector mixture.

#### Mathematical Form

**Representation space:** Direct activation space $z = h$ (or optionally $z = \phi_\theta(h)$ with a lightweight encoder).

**Local projector family:**
Given $K$ clusters of benign activations with centroids $\{\mu_k\}_{k=1}^K$:

$$P_k = Q_k Q_k^\top, \quad Q_k = \text{NullSpace}(H_{b,k})$$

where $H_{b,k}$ is the benign activation matrix for cluster $k$.

**Kernel gating:**

$$w_k(h) = \frac{\exp(-\|h - \mu_k\|^2 / 2\sigma_k^2)}{\sum_{j=1}^K \exp(-\|h - \mu_j\|^2 / 2\sigma_j^2)}$$

**Local steering matrix:**

$$M(h) = \sum_{k=1}^K w_k(h) \cdot P_k \tilde{\Delta}_k$$

**Update rule:**

$$h' = h + \lambda \cdot (h \cdot M(h))$$

#### Training Objective

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{benign}}}_{\text{utility preservation}} + \alpha \underbrace{\mathcal{L}_{\text{safety}}}_{\text{safety enhancement}} + \beta \underbrace{\mathcal{L}_{\text{local}}}_{\text{local consistency}}$$

where:

- $\mathcal{L}_{\text{benign}} = \frac{1}{|B|}\sum_{h_b \in B} \|h_b \cdot M(h_b)\|^2$ — benign activations get near-zero steering
- $\mathcal{L}_{\text{safety}} = \frac{1}{|H|}\sum_{h_m \in H} \|h_m \cdot M(h_m) - r_l\|^2$ — malicious activations steer toward refusal vector $r_l$
- $\mathcal{L}_{\text{local}} = \frac{1}{|B|}\sum_{h_b \in B} \sum_{h' \in \mathcal{N}(h_b)} \|M(h_b) - M(h')\|_F^2$ — local smoothness of steering matrix

**Trainable parameters:** Cluster centroids $\{\mu_k\}$, bandwidths $\{\sigma_k\}$, per-cluster $\tilde{\Delta}_k$.

**Initialization:** $K$-means on benign activations → SVD per cluster → AlphaSteer-style $\tilde{\Delta}_k$ solve.

#### Inference

```
For each steered layer l:
    1. Extract last-token hidden state h
    2. Compute kernel weights w_k(h) for k = 1..K
    3. Compute M(h) = Σ_k w_k(h) · P_k · Δ̃_k
    4. Apply: h' = h + λ · (h · M(h))
```

#### Local Point Invariance

- **At benign anchors:** $w_k(h_b)$ concentrates on the cluster $k$ that contains $h_b$, and $P_k$ is the null-space of that cluster → $h_b \cdot P_k \tilde{\Delta}_k \approx 0$
- **Near benign anchors:** kernel smoothness ensures $M(h) \approx M(h_b) \approx 0$ for $h \in \mathcal{N}_\epsilon(h_b)$
- **At malicious states:** $w_k(h_m)$ may distribute across clusters or concentrate on none → $M(h_m)$ is not null-space constrained → steering toward refusal

#### Relation to AlphaSteer

When $K=1$, $w_1(h) = 1$ for all $h$, $P_1 = \hat{P}$, and $\tilde{\Delta}_1 = \tilde{\Delta}$. The method reduces exactly to AlphaSteer: $M(h) = \hat{P}\tilde{\Delta}$.

---

### Candidate A2: **ContrastiveAlphaSteer** (Contrastive-AE + Local Null-Space)

**Type:** Curveball-style in a learned latent space

**Core idea:** Train a lightweight autoencoder $\phi_\theta: \mathcal{H} \to \mathcal{Z}$ and $\psi_\theta: \mathcal{Z} \to \mathcal{H}$ such that the latent space $\mathcal{Z}$ is locally smooth for benign activations and separates malicious activations. Apply AlphaSteer-style null-space steering in $\mathcal{Z}$.

#### Mathematical Form

**Encoder/Decoder:**
$$z = \phi_\theta(h), \quad \hat{h} = \psi_\theta(z)$$

**Latent steering:**
$$z' = z + \lambda \cdot (z \cdot \hat{P}_z \tilde{\Delta}_z)$$

where $\hat{P}_z$ is the null-space projector in latent space.

**Back to activation space:**
$$h' = \psi_\theta(z')$$

#### Training Objective

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \alpha \mathcal{L}_{\text{contrastive}} + \beta \mathcal{L}_{\text{Jacobian}} + \gamma \mathcal{L}_{\text{steering}}$$

where:
- $\mathcal{L}_{\text{recon}} = \|h - \psi_\theta(\phi_\theta(h))\|^2$
- $\mathcal{L}_{\text{contrastive}}$ = triplet loss separating benign/malicious in $\mathcal{Z}$
- $\mathcal{L}_{\text{Jacobian}} = \|\nabla_h \phi_\theta(h_b)\|_F^2$ at benign anchors — local flatness
- $\mathcal{L}_{\text{steering}}$ = AlphaSteer objectives in latent space

#### Local Point Invariance

Jacobian regularization at benign anchors ensures $\phi_\theta$ is locally flat → small perturbations of $h_b$ map to nearly the same $z$ → steering in $\mathcal{Z}$ produces near-zero change → $h' \approx h$.

#### Relation to AlphaSteer

When $\phi_\theta = \psi_\theta = \text{Id}$ (identity), the method reduces to AlphaSteer in the original space. The autoencoder adds a learned change-of-basis that can make the null-space more locally accurate.

---

### Candidate B1: **FieldAlphaSteer** (Position-Dependent Steering Field)

**Type:** ODESteer-style (direction field in activation space)

**Core idea:** Replace the fixed steering matrix with a learned steering field $f_\theta(h)$ that maps each activation to a steering vector. The field is constrained to be near-zero at benign anchors and aligned with refusal directions at malicious states.

#### Mathematical Form

**Steering field:**
$$f_\theta(h) = g_\theta(h) \odot d_{\text{refusal}}$$

where $g_\theta(h): \mathcal{H} \to \mathbb{R}$ is a learned scalar gate (or low-rank projection).

**Update rule:**
$$h' = h + \lambda \cdot f_\theta(h)$$

#### Training Objective

$$\mathcal{L} = \underbrace{\sum_{h_b} \|f_\theta(h_b)\|^2}_{\text{zero at benign}} + \alpha \underbrace{\sum_{h_m} \|f_\theta(h_m) - r_l\|^2}_{\text{refusal at malicious}} + \beta \underbrace{\sum_{h_b} \|\nabla_h f_\theta(h_b)\|_F^2}_{\text{local smoothness}}$$

#### Local Point Invariance

The training loss directly enforces $f_\theta(h_b) = 0$ at benign points and $\|\nabla_h f_\theta(h_b)\| \approx 0$ ensures smooth decay near benign points.

#### Relation to AlphaSteer

This is a more expressive generalization. AlphaSteer's $h \cdot M$ is a special case where $f_\theta(h) = h \cdot M$ is a fixed linear function. FieldAlphaSteer replaces this with a learned nonlinear function.

---

## 6. Phase 3: Comparison and Final Recommendation

### T3.1 Scoring Matrix

| Dimension | A1: LocalAlphaSteer | A2: ContrastiveAlphaSteer | B1: FieldAlphaSteer |
|-----------|:---:|:---:|:---:|
| 1. Single-space elegance | **5** (direct activation space) | 4 (latent space is single but adds AE) | **5** (direct activation space) |
| 2. AlphaSteer narrative continuity | **5** ($K=1$ degenerates exactly) | 3 (AE is a new concept) | 3 (field replaces matrix) |
| 3. Local point invariance | **5** (kernel-gated null-space) | 4 (Jacobian reg) | 4 (loss-based) |
| 4. Nonlinear steering compatibility | **4** (locally linear, globally nonlinear) | 4 (nonlinear via AE) | **5** (fully nonlinear field) |
| 5. Training/inference complexity | **5** (SVD init + lightweight fine-tune) | 3 (full AE training) | 4 (MLP training, simple inference) |
| 6. P1/P3/P5 verification convenience | **5** (direct comparison with AlphaSteer) | 3 (confounded by AE quality) | 4 (can ablate field complexity) |
| **Total** | **29** | **21** | **25** |

### T3.2 Final Recommendation

> **If implementation starts now, the first priority is Candidate A1: LocalAlphaSteer (Kernel-Weighted Local Projectors).**

**Rationale:**
1. **Most natural AlphaSteer upgrade:** The global projector $\hat{P}$ becomes a family $\{P_k\}$ with soft gating — this is a minimal conceptual extension.
2. **$K=1$ degeneracy:** Provides a clean ablation path back to AlphaSteer.
3. **SVD initialization:** The existing AlphaSteer pipeline (cluster → SVD → solve) maps directly to LocalAlphaSteer (K-means → per-cluster SVD → per-cluster solve → fine-tune gates).
4. **Lowest training overhead:** Only need to fine-tune $\{\mu_k, \sigma_k, \tilde{\Delta}_k\}$, not an entire autoencoder or field network.
5. **Direct P2/P4 validation:** Can measure whether K local projectors reduce the "global projection compromise" that P4 describes.

### T3.3 Rejection Rationale for Other Candidates

**A2 (ContrastiveAlphaSteer) — Not first choice because:**
- Introduces an autoencoder, which is a large new component with its own failure modes
- Reconstruction loss may conflict with steering objectives
- Training cost is significantly higher
- Harder to attribute improvements: is it the AE or the local constraint?

**B1 (FieldAlphaSteer) — Not first choice because:**
- Breaks the "matrix multiplication" narrative of AlphaSteer
- The learned field $f_\theta$ is a black box — harder to interpret why benign invariance holds
- More parameters, slower inference
- Does not naturally degenerate to AlphaSteer (no $K=1$ reduction)

Both are viable backup plans if LocalAlphaSteer fails systematically.

---

## 7. Phase 4: Mathematical Draft of LocalAlphaSteer

### T4.1 Representation Space

**Space definition:**
$$z = h \in \mathbb{R}^d$$

where $h$ is the hidden state at the last input token at a given transformer layer. No learned encoder is needed — steering operates directly in the activation space.

**Why single space:** Both the benign constraint (null-space of local benign clusters) and the steering action (refusal direction projection) operate on the same $\mathbb{R}^d$.

**Why suitable for local constraints:** The activation space naturally supports kernel-based locality — nearby activations correspond to semantically similar inputs.

**Why suitable for nonlinear steering:** The kernel-weighted mixture of linear projectors is globally nonlinear while being locally linear — a piecewise-linear approximation of the ideal local steering.

### T4.2 Local Benign Constraint

#### Anchor Construction

Given benign activations $\{h_b^{(i)}\}_{i=1}^N$ at layer $l$:

1. **Clustering:** Run $K$-means on $\{h_b^{(i)}\}$ to obtain $K$ clusters with centroids $\{\mu_k\}_{k=1}^K$ and assignments $\{C_k\}_{k=1}^K$.

2. **Per-cluster null-space:** For each cluster $k$, compute the local benign activation matrix:
$$H_{b,k} = [h_b^{(i)}]_{i \in C_k} \in \mathbb{R}^{|C_k| \times d}$$
and its null-space projector:
$$Q_k = \text{NullSpace}(H_{b,k}), \quad P_k = Q_k Q_k^\top \in \mathbb{R}^{d \times d}$$

3. **Bandwidth estimation:** Set $\sigma_k$ as the average distance of cluster members to centroid:
$$\sigma_k = \frac{1}{|C_k|} \sum_{i \in C_k} \|h_b^{(i)} - \mu_k\|$$

#### Kernel Gating

For an input activation $h$, compute soft assignment weights:
$$w_k(h) = \frac{\exp\bigl(-\|h - \mu_k\|^2 / (2\sigma_k^2)\bigr)}{\sum_{j=1}^K \exp\bigl(-\|h - \mu_j\|^2 / (2\sigma_j^2)\bigr)}$$

#### Local Point Invariance

The local point invariance property is achieved through:

1. **Point invariance at anchors:**
For $h_b \in C_k$: $w_k(h_b) \approx 1$ and $h_b \in \text{row}(H_{b,k})$ implies $h_b \cdot P_k \approx 0$, hence:
$$h_b \cdot M(h_b) = h_b \cdot \sum_j w_j(h_b) P_j \tilde{\Delta}_j \approx h_b \cdot P_k \tilde{\Delta}_k \approx 0$$

2. **Neighborhood approximate invariance:**
For $h \in \mathcal{N}_\epsilon(h_b)$: by kernel smoothness, $w_k(h) \approx w_k(h_b)$ and $h \cdot P_k \approx h_b \cdot P_k \approx 0$, so $M(h) \approx M(h_b) \approx 0$.

3. **Malicious non-invariance:**
For $h_m$ far from all benign clusters: kernel weights are diffuse, and $h_m$ is NOT in the row space of any $H_{b,k}$, so $h_m \cdot P_k \neq 0$ → steering is active.

This can optionally be reinforced by a training loss:

$$\mathcal{L}_{\text{local-inv}} = \frac{1}{|B|}\sum_{h_b \in B} \|h_b \cdot M(h_b)\|^2 + \gamma \sum_{h_b \in B} \sum_{h' \in \mathcal{N}(h_b)} \|M(h_b) - M(h')\|_F^2$$

### T4.3 Steering Definition

**Type:** Curveball-style with locally-varying direction.

**Composite steering matrix:**
$$M(h) = \sum_{k=1}^K w_k(h) \cdot P_k \tilde{\Delta}_k \in \mathbb{R}^{d \times d}$$

**Steering update:**
$$h' = h + \lambda \cdot (h \cdot M(h))$$

where $\lambda$ is the layer-specific strength coefficient (same role as in AlphaSteer).

**Per-cluster refusal alignment:**
Each $\tilde{\Delta}_k$ is solved so that harmful activations within the influence region of cluster $k$ are steered toward the refusal vector:

$$\tilde{\Delta}_k = \arg\min_{\Delta} \|H_{h,k} P_k \Delta - r_l \mathbf{1}^\top\|_F^2 + \lambda_{\text{reg}} \|P_k^\top P_k \Delta\|_F^2$$

where $H_{h,k}$ are harmful activations weighted by $w_k$.

### T4.4 Training Objective

**Full training loss:**

$$\mathcal{L} = \mathcal{L}_{\text{benign}} + \alpha \mathcal{L}_{\text{safety}} + \beta \mathcal{L}_{\text{smooth}} + \gamma \mathcal{L}_{\text{reg}}$$

**Component definitions:**

1. **Benign preservation (utility preservation):**
$$\mathcal{L}_{\text{benign}} = \frac{1}{|B|}\sum_{h_b \in B} \|h_b \cdot M(h_b)\|^2$$

2. **Safety enhancement:**
$$\mathcal{L}_{\text{safety}} = \frac{1}{|H|}\sum_{h_m \in H} \|h_m \cdot M(h_m) - r_l\|^2$$

3. **Local smoothness (neighborhood consistency):**
$$\mathcal{L}_{\text{smooth}} = \frac{1}{|B|}\sum_{h_b \in B} \frac{1}{|\mathcal{N}(h_b)|}\sum_{h' \in \mathcal{N}(h_b)} \|M(h_b) - M(h')\|_F^2$$

4. **Regularization (prevent degenerate projectors):**
$$\mathcal{L}_{\text{reg}} = \sum_{k=1}^K \|\tilde{\Delta}_k\|_F^2$$

**Training procedure:**

```
Phase 1 (Initialization):
    1. K-means on H_b → centroids {μ_k}, assignments {C_k}
    2. For each k: SVD-based null-space P_k = Q_k Q_k^T
    3. For each k: solve Δ̃_k via regularized pseudoinverse
    4. Estimate σ_k from cluster spread

Phase 2 (Fine-tuning):
    Trainable: {μ_k, σ_k, Δ̃_k} for k = 1..K
    Fixed: {P_k} (recomputed periodically or kept fixed)
    Optimizer: Adam, lr=1e-4, weight_decay=1e-5
    Epochs: 50–100 with early stopping on validation
```

**Inference procedure:**

```
Input: hidden state h at layer l, strength λ
Preloaded: {μ_k, σ_k, P_k, Δ̃_k} for k = 1..K

1. Compute w_k(h) for all k (K kernel evaluations + softmax)
2. Compute M(h) = Σ_k w_k(h) · (P_k · Δ̃_k)  [can precompute P_k Δ̃_k]
3. steering_vector = h · M(h) · λ
4. Return h' = h + steering_vector (broadcast to all positions)
```

**Inference optimization:** Precompute $S_k = P_k \tilde{\Delta}_k$ once after training. At inference, only kernel weights need to be computed dynamically:

$$M(h) = \sum_k w_k(h) S_k$$

This reduces inference cost to: $K$ distance computations + softmax + weighted sum of $K$ matrices × vector.

### Variable Summary

| Symbol | Shape | Description |
|--------|-------|-------------|
| $h$ | $(d,)$ | Hidden state at last input token |
| $K$ | scalar | Number of local projector clusters |
| $\mu_k$ | $(d,)$ | Centroid of benign cluster $k$ |
| $\sigma_k$ | scalar | Bandwidth of cluster $k$ |
| $w_k(h)$ | scalar | Kernel gate weight for cluster $k$ |
| $H_{b,k}$ | $(n_k, d)$ | Benign activations in cluster $k$ |
| $Q_k$ | $(d, d_k)$ | Null-space basis of cluster $k$ |
| $P_k$ | $(d, d)$ | Null-space projector of cluster $k$ |
| $\tilde{\Delta}_k$ | $(d, d)$ | Per-cluster steering solution |
| $S_k = P_k\tilde{\Delta}_k$ | $(d, d)$ | Precomputed steering matrix for cluster $k$ |
| $M(h)$ | $(d, d)$ | Position-dependent composite steering matrix |
| $r_l$ | $(d,)$ | Refusal vector at layer $l$ |
| $\lambda$ | scalar | Steering strength |
| $\alpha, \beta, \gamma$ | scalar | Loss balancing coefficients |

---

## 8. Phase 5: Experiment Plan

### T5.1 Connection with Existing Experiments

**From current G1/G2 / 1.5A results:**

- **P2 is strongly supported** → Local projector family is a real phenomenon → LocalAlphaSteer directly addresses this
- **P4's local distortion is supported** → Global projector compromises → K>1 local projectors should reduce distortion
- **PCA/kernel PCA are better for benign charts** → Kernel-based locality is the right inductive bias
- **RFF works for malicious separation in mid-layers** → Validates kernel methods for this task

**Directly reusable:**
- Benign/malicious embeddings (already extracted)
- Refusal vectors (already computed)
- AlphaSteer baselines (already run)
- Evaluation pipeline (jailbreak.py, xstest.py, alpaca.py)

**New additions:**
- K-means clustering of benign activations
- Per-cluster null-space computation
- Kernel-gated inference module
- Local violation diagnostics

### T5.2 Minimum Viable Experiment (MVP)

**MVP Goal:** Demonstrate that LocalAlphaSteer with $K > 1$ reduces benign local violation compared to AlphaSteer ($K=1$) without degrading safety.

#### Experiment 1: Local Constraint Geometry
**Question:** Do K local projectors better fit the benign activation geometry than 1 global projector?

**Protocol:**
1. Cluster benign activations into $K \in \{1, 2, 4, 8, 16\}$ clusters
2. Compute per-cluster null-space projectors
3. Measure residual: $\text{BenignResidual}(K) = \frac{1}{N}\sum_i \|h_b^{(i)} \cdot P_{k(i)}\|^2$
4. Compare with global residual: $\text{BenignResidual}(1) = \frac{1}{N}\sum_i \|h_b^{(i)} \cdot \hat{P}\|^2$

**Expected:** $\text{BenignResidual}(K) < \text{BenignResidual}(1)$ for $K > 1$

#### Experiment 2: Touches P1 (Local Incompatibility)
**Question:** Does local projection reduce the safety–utility trade-off?

**Protocol:**
1. Run LocalAlphaSteer with $K \in \{1, 4, 8\}$ on jailbreak test sets
2. Measure: Attack Success Rate (ASR) and AlpacaEval score
3. Compare Pareto fronts across $K$ values

**Expected:** $K > 1$ should improve the Pareto front (better safety at same utility, or better utility at same safety)

#### Experiment 3: Touches P3 (Projection Superiority)
**Question:** Is the locally-projected steering direction more compatible with the ideal local direction?

**Protocol:**
1. At each test point, compute:
   - Global projected direction: $d_{\text{global}} = h \cdot \hat{P}\tilde{\Delta}$
   - Local projected direction: $d_{\text{local}} = h \cdot M(h)$
   - Oracle direction: direction that maximally increases refusal probability (estimated via gradient)
2. Measure directional compatibility: $\mu(h) = \cos(d_{\text{local}}, d_{\text{oracle}})$

**Expected:** $\mu_{\text{local}} > \mu_{\text{global}}$ on average, especially for hard cases

### T5.3 Key Diagnostic Metrics

| Metric | Symbol | Definition | Purpose |
|--------|--------|-----------|---------|
| Direction compatibility | $\mu(z)$ | $\cos(d_{\text{local}}, d_{\text{oracle}})$ | Measures alignment with ideal direction |
| Benign local violation | BLV | $\frac{1}{N_b}\sum\|h_b \cdot M(h_b)\|^2$ | Should be ≈ 0 |
| Safety steering magnitude | SSM | $\frac{1}{N_m}\sum\|h_m \cdot M(h_m)\|$ | Should be large |
| Projected vs unprojected gap | PUG | $\|\text{projected} - \text{unprojected}\|$ | Measures constraint cost |
| Kernel concentration | KC | $\max_k w_k(h)$ | Measures soft vs hard assignment |
| K-sensitivity | KS | $\partial(\text{ASR}) / \partial K$ | Diminishing returns of more clusters |
| Hard-case improvement | HCI | ASR improvement on hardest 10% of attacks | Validates local adaptation |

### T5.4 Success / Failure Criteria

#### Success Criteria (supports LocalAlphaSteer as main line)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| BLV decreases with $K$ | BLV($K$=8) < 0.5 × BLV($K$=1) | Local projectors are geometrically meaningful |
| ASR improves at same utility | ASR($K$=8) ≤ ASR($K$=1) at same AlpacaEval | Local adaptation helps safety |
| $\mu_{\text{local}} > \mu_{\text{global}}$ | Average improvement ≥ 0.05 cosine | P3 is supported |
| KC is moderate | Avg $\max_k w_k \in [0.5, 0.9]$ | Soft gating is active (not collapsing to K=1) |

#### Failure Criteria (single-space at risk)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| BLV does NOT decrease with $K$ | BLV($K$=8) ≥ 0.9 × BLV($K$=1) | Local projectors don't help geometry |
| Pareto front does NOT improve | No improvement for any $K$ | Trade-off is fundamental, not geometric |
| KC collapses | Avg $\max_k w_k > 0.95$ for all $K$ | Soft gating degenerates to hard; single cluster dominates |
| Training instability | Loss diverges for $K > 4$ | Optimization landscape is problematic |

---

## 9. Phase 6: Single-Space Failure Criteria and Dual-Space Backup

### T6.1 Single-Space Failure Standards

The single-space approach (LocalAlphaSteer) should be considered failed if **two or more** of the following occur:

1. **Chart–separation incompatibility:** $K$ local projectors with the best null-space ratio consistently fail to simultaneously achieve BLV < threshold AND maintain safety steering magnitude > threshold. This means benign charting and malicious separation CANNOT coexist in the same space geometry.

2. **Persistent local invariance–safety trade-off:** Even with optimal $K$ and $\sigma_k$, improving benign local invariance (lower BLV) always degrades safety steering magnitude (lower SSM) by a proportional amount. This indicates P1 (local incompatibility) is fundamental, not solvable by local projectors.

3. **Projected frontier dominated:** A simple dual-space baseline (separate chart space + separation space) achieves strictly better Pareto frontier than LocalAlphaSteer for ALL $K$ values tested. This means the single-space constraint is genuinely limiting.

4. **Complexity ceiling:** Achieving competitive results requires $K > 32$ or $\sigma_k$ tuning per layer per dataset, making the method no simpler than explicit dual-space approaches.

### T6.2 Minimal Dual-Space Backup

If single-space fails, the minimal dual-space version preserves as much of the LocalAlphaSteer narrative as possible:

**Architecture:**
```
Shared encoder: φ_θ(h) → z ∈ R^d_z
Chart head:     π_chart(z) → local benign assignment weights w_k
Separation head: π_sep(z) → malicious steering gate g(z) ∈ [0, 1]
```

**Steering in original space:**
$$h' = h + \lambda \cdot g(\phi_\theta(h)) \cdot \sum_k w_k(\phi_\theta(h)) \cdot S_k \cdot h$$

**Key properties preserved:**
- Steering acts on real activation space $\mathcal{H}$
- $K=1$ and $g \equiv 1$ degenerates to AlphaSteer
- Training is end-to-end
- Inference is plug-and-play (encoder is a small MLP)

**Key difference from single-space:**
- The chart weights $w_k$ and separation gate $g$ are computed in a learned latent space $\mathcal{Z}$, not in the raw activation space $\mathcal{H}$
- This allows the two geometric roles to use different feature representations while keeping the final steering in a single action space

---

## 10. Implementation Roadmap

### Week 1: Infrastructure

- [ ] Implement `local_steering_utils.py`: K-means clustering, per-cluster null-space, kernel gating
- [ ] Implement `LocalAlphaLlama.py`: Decoder layer with K local projectors
- [ ] Unit tests for local steering utilities
- [ ] Validate K=1 degeneracy matches AlphaSteer exactly

### Week 2: Baseline Experiments

- [ ] Run Experiment 1 (Local Constraint Geometry) for $K \in \{1, 2, 4, 8, 16\}$
- [ ] Measure BLV across layers and $K$ values
- [ ] Visualize kernel concentration and cluster assignments
- [ ] Decision gate: if BLV does not decrease with $K$, investigate before proceeding

### Week 3: Safety–Utility Experiments

- [ ] Run Experiment 2 (P1) with full evaluation pipeline
- [ ] Compare Pareto fronts across $K$ values
- [ ] Measure HCI on hardest attack categories (GCG, AutoDAN)
- [ ] Fine-tune $\{\mu_k, \sigma_k, \tilde{\Delta}_k\}$ end-to-end

### Week 4: Direction Quality and Write-up

- [ ] Run Experiment 3 (P3) with directional compatibility metrics
- [ ] Compare local vs global projected directions
- [ ] Compile results into paper-ready figures
- [ ] Decision gate: determine if single-space is viable or dual-space backup needed
- [ ] Draft method section for paper

---

*Document version: 1.0*
*Last updated: 2026-03-30*
