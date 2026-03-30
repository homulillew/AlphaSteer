# Next-Generation AlphaSteer: From Global Linear Null-Space to Locally-Gated Nonlinear Steering

**Research Proposal — Single-Round Jailbreak Defense via Local Activation Steering**

---

## 1. Executive Summary

AlphaSteer's core contribution was rewriting activation steering as a **constrained learning problem**: the steering matrix $\Delta$ is trained subject to a null-space constraint $\Delta H_b \approx 0$, ensuring near-zero steering on benign prompts while projecting harmful activations toward a refusal direction. However, our experiments (G1/G2, Phase 1.5A) have conclusively shown that the "benign constraint = a single global linear null-space" assumption is **fundamentally insufficient**—local projectors vary significantly across the activation manifold, near-neighbor projectors are more similar than distant ones, and global projector approximation error is large.

This proposal presents **LocalAlphaSteer**, a principled upgrade that replaces the global linear null-space with a **locally-gated projector family** in a **single learned space**. The method:

1. Clusters benign activations into $K$ anchor neighborhoods
2. Computes **local null-space projectors** $P_k$ at each anchor
3. Learns **soft kernel gating weights** $w_k(h)$ that blend local projectors based on activation position
4. Solves for **local steering directions** $\delta_k$ at each anchor via regularized regression
5. Produces a **position-dependent steering matrix**: $\Delta(h) = \sum_k w_k(h) \cdot P_k \cdot \delta_k$

This preserves AlphaSteer's key virtues (trained, interpretable, constraint-internalized, inference-time pluggable) while addressing the empirically established insufficiency of global constraints.

---

## 2. Precise Problem Restatement

### What AlphaSteer solved
AlphaSteer formalized activation steering as a dual-objective optimization:
- **Utility preservation**: $\Delta H_b \approx 0$ (near-zero steering on benign prompts)
- **Safety enhancement**: $H_h \Delta \approx r$ (reconstruct refusal direction on harmful prompts)

It achieved this by computing a global null-space projection $\hat{P}$ from benign activations via SVD, then solving $\Delta = \tilde{\Delta} \hat{P}$.

### What experiments have proven insufficient
1. **P2 (strongly supported)**: The benign constraint is not a single fixed null-space but a **family of local subspaces** that vary smoothly across the activation manifold.
2. **P4 (partially supported)**: The global projector $\hat{P}$ introduces **significant local distortion**—it over-constrains some regions and under-constrains others.
3. **P6 (refined)**: Different representation spaces have different **geometric visibility**—PCA/kernel-PCA favor benign charting; RFF favors malicious separation—but this does not automatically preclude a single-space solution.

### What we need
A method that:
- Replaces the **global fixed null-space** with a **local projector family**
- Maintains a **single space** for both benign charting and malicious separation (preferred for narrative elegance)
- Achieves **local point invariance** (near-zero steering at benign anchor points) and **local neighborhood approximate invariance** (smooth decay around anchors)
- Remains **trainable, interpretable, and inference-time pluggable** like AlphaSteer
- Serves as a **natural upgrade** of AlphaSteer, not a separate method

---

## 3. Theoretical Upgrade Path: AlphaSteer → LocalAlphaSteer

### 3.1 The Progression

| Aspect | AlphaSteer | LocalAlphaSteer |
|--------|-----------|-----------------|
| Constraint | Global null-space $\hat{P}$ | Local projector family $\{P_k\}_{k=1}^K$ |
| Controller | Fixed matrix $\Delta$ | Position-dependent $\Delta(h)$ |
| Benign guarantee | $\Delta H_b \approx 0$ globally | $\Delta(h_b) \cdot h_b \approx 0$ locally per anchor |
| Safety mechanism | Global regression $H_h \Delta \approx r$ | Local regression at each anchor |
| Space | Original activation space | Same space, locally structured |
| Training | SVD + pseudoinverse (closed-form) | Clustering + local SVD + kernel gating (hybrid) |

### 3.2 Why This Is a Natural Upgrade

AlphaSteer's deepest insight was: **"put the constraint inside the steering matrix itself."** LocalAlphaSteer preserves exactly this philosophy—it still internalizes the benign constraint into the steering operator. The only change is that the constraint object evolves from a single global subspace to a family of local subspaces, exactly as the experimental evidence demands.

Mathematically:
- **AlphaSteer**: $s(h) = (\tilde{\Delta} \hat{P}) \cdot h = \Delta \cdot h$
- **LocalAlphaSteer**: $s(h) = \sum_k w_k(h) \cdot (P_k \tilde{\delta}_k) \cdot h = \Delta(h) \cdot h$

When $K = 1$ and $w_1(h) = 1$ everywhere, LocalAlphaSteer reduces exactly to AlphaSteer.

### 3.3 Key Design Principles Inherited from AlphaSteer

1. **Constraint internalization**: The local projectors $P_k$ are baked into the steering operator, not applied as post-hoc filters.
2. **Trainability**: All components ($\{z_k, P_k, \delta_k\}$) are computed from data.
3. **Interpretability**: Each anchor $z_k$ represents a recognizable region of activation space; $P_k$ is its local null-space; $\delta_k$ is the local refusal direction.
4. **Pluggability**: At inference, the locally-gated steering matrix operates identically to AlphaSteer—it's just a function $h \mapsto \Delta(h) \cdot h$ applied at each decoder layer.

---

## 4. Single-Space Candidate Method Panorama

### Candidate 1: Locally-Gated Projector Network (LGP) ⭐ **RECOMMENDED**

**Core idea**: Partition benign activation space into $K$ soft neighborhoods via kernel-weighted anchors; compute a local null-space projector and local steering direction per anchor; blend them with position-dependent soft gating.

**Mathematical formulation**:

Given benign anchor points $\{z_k\}_{k=1}^K$ (from k-means on benign activations), define:

$$w_k(h) = \frac{\exp(-\|h - z_k\|^2 / \tau)}{\sum_{j=1}^K \exp(-\|h - z_j\|^2 / \tau)}$$

At each anchor, compute the local projector from nearby benign activations:

$$P_k = Q_k Q_k^\top, \quad Q_k = \text{null-space-basis}(H_b^{(k)})$$

where $H_b^{(k)} = \{h_b : w_k(h_b) > \theta\}$ is the set of benign activations in the $k$-th neighborhood.

Solve for the local steering direction at each anchor:

$$\delta_k = \arg\min_\delta \|H_h^{(k)} P_k \delta - r\|^2 + \lambda \|P_k \delta\|^2$$

The final position-dependent steering is:

$$s(h) = \lambda_{\text{strength}} \cdot \left(\sum_{k=1}^K w_k(h) \cdot P_k \delta_k\right)^\top h$$

**Relation to AlphaSteer**: Direct generalization. When $K=1$, $w_1=1$, this is AlphaSteer. The null-space constraint is local rather than global but the mathematical structure is identical.

**Why it fits our experimental evidence**:
- P2 (local subspace family): Each $P_k$ captures the local null-space at anchor $z_k$.
- P4 (global projector distortion): By construction, each $P_k$ is accurate for its local neighborhood, eliminating global distortion.
- P6 (geometric visibility): Operates in original activation space, but locally linearizes via the anchor decomposition—PCA-like local charting with kernel-smoothed blending.

**Risk points**:
- Choice of $K$ and $\tau$ requires tuning (use validation set).
- Very sparse neighborhoods may give unstable local SVDs (mitigate with regularization and minimum neighbor count).
- Memory: storing $K$ projector matrices per layer ($K \times d \times d$); mitigate with low-rank approximation.

**Landing difficulty**: Low-Medium. Uses same mathematical tools as AlphaSteer (SVD, pseudoinverse) plus k-means clustering and softmax blending.

**Priority**: ★★★★★ (Highest)

---

### Candidate 2: Learned Contractive Encoder + Projected Steering (LCE)

**Core idea**: Learn a nonlinear encoder $\phi_\theta(h)$ with a contractive (Jacobian-penalized) loss that makes benign activations locally flat in the encoded space, then perform AlphaSteer-style null-space projection in the learned space and decode back.

**Mathematical formulation**:

Train encoder $\phi_\theta: \mathbb{R}^d \to \mathbb{R}^m$ and decoder $\psi_\theta: \mathbb{R}^m \to \mathbb{R}^d$ with loss:

$$\mathcal{L} = \underbrace{\|h - \psi_\theta(\phi_\theta(h))\|^2}_{\text{reconstruction}} + \alpha \underbrace{\|J_\phi(h_b)\|_F^2}_{\text{Jacobian contraction on benign}} + \beta \underbrace{\mathcal{L}_{\text{separation}}(\phi_\theta(h_b), \phi_\theta(h_m))}_{\text{benign-malicious separation}}$$

In the learned space, compute a single (or local) null-space projector:

$$\hat{P}_\phi = \text{null-space-proj}(\phi_\theta(H_b))$$

Steering in learned space: $\tilde{s}(\phi(h)) = \hat{P}_\phi \cdot \tilde{\delta}$, then decode back.

**Relation to AlphaSteer**: Replaces the raw activation space with a learned space optimized for local flatness + separation, then applies AlphaSteer's projector logic in that space.

**Why it fits**:
- The Jacobian penalty flattens benign activations locally → better local charting.
- The separation term pushes malicious activations apart → better anomaly detection.
- If the space is well-learned, a single projector might suffice (restoring the simplicity of AlphaSteer but in a better space).

**Risk points**:
- Autoencoder training adds significant complexity.
- The decode step must preserve the model's internal representations faithfully.
- May require separate autoencoders per layer.
- If the learned space still doesn't admit a good global projector, we've added complexity without solving the core problem.

**Landing difficulty**: Medium-High. Requires training autoencoders, designing the multi-term loss, and integrating with the decoder pathway.

**Priority**: ★★★☆☆ (Backup if LGP's local projectors don't yield sufficient smoothness)

---

### Candidate 3: Piecewise-Linear Expert Steering (PLE)

**Core idea**: Frame the steering problem as a Mixture-of-Experts, where each "expert" is a local linear steering matrix, and a learned gating network routes each activation to the appropriate expert(s).

**Mathematical formulation**:

Define $K$ expert steering matrices $\{M_k\}_{k=1}^K$ and a gating network $g_\theta: \mathbb{R}^d \to \Delta^K$:

$$s(h) = \lambda \cdot \left(\sum_{k=1}^K g_k(h) \cdot M_k\right) h$$

The gating network is a small MLP: $g_\theta(h) = \text{softmax}(W_2 \cdot \text{ReLU}(W_1 h + b_1) + b_2)$.

Training loss:

$$\mathcal{L} = \underbrace{\sum_{h_b} \|s(h_b)\|^2}_{\text{benign near-zero}} + \underbrace{\sum_{h_m} \|s(h_m) - r\|^2}_{\text{malicious → refusal}} + \gamma \underbrace{\sum_k \text{rank}(M_k)}_{\text{low-rank regularization}}$$

**Relation to AlphaSteer**: AlphaSteer is the $K=1$ case without gating. This generalizes to position-dependent linear steering via soft expert selection.

**Why it fits**:
- Naturally implements position-dependent steering.
- Each expert can specialize: some for "clearly benign" regions (zero matrix), some for "clearly harmful" (refusal-aligned matrix), and boundary experts for nuanced cases.
- The gating network is learned end-to-end, adapting to the actual geometry.

**Risk points**:
- Loses the explicit null-space constraint—the benign near-zero property is enforced only via the loss, not structurally.
- End-to-end training may be harder to interpret than AlphaSteer's closed-form solution.
- Risk of overfitting with too many experts.

**Landing difficulty**: Medium. Standard MoE implementation with custom loss.

**Priority**: ★★★★☆ (Strong backup; less principled than LGP but more flexible)

---

### Candidate 4 (Dual-Space Backup): Charting-Space + Separation-Space (CSS)

**Core idea**: Accept that a single space cannot simultaneously optimize for benign charting and malicious separation. Use two specialized spaces: a charting space $\phi_c(h)$ (PCA/kernel-PCA-like) for benign constraint computation, and a separation space $\phi_s(h)$ (RFF-like) for malicious detection and steering direction computation. Both spaces share the final steering action in activation space.

**Mathematical formulation**:

1. **Charting space**: $\phi_c(h) = V_{\text{PCA}}^\top h$ (local PCA basis) → Compute local projectors $P_k^c$ in this space.
2. **Separation space**: $\phi_s(h) = \text{RFF}(h)$ → Compute safety field direction $g^s(h)$.
3. **Combined steering**: $s(h) = \lambda \cdot \text{decode}(P^c_{\text{local}}(h) \cdot g^s(h))$

**Relation to AlphaSteer**: Both spaces inform the constraint and direction, but the final steering still acts in the model's activation space. It's AlphaSteer with specialized "lenses" for different roles.

**Risk points**: More complex narrative; harder to explain why two spaces are needed vs. one well-designed space; potential alignment issues between spaces.

**Landing difficulty**: High. Requires careful design of inter-space communication.

**Priority**: ★★☆☆☆ (Only if single-space approaches systematically fail)

---

### Candidate 5 (Dual-Space Backup): Shared-Encoder Dual-Head (SEDH)

**Core idea**: Train a single shared encoder $\phi_\theta(h)$ but with two heads—a "charting head" $\phi_c$ and a "separation head" $\phi_s$—with different regularization objectives. The shared encoder ensures coherence; the heads specialize.

**Mathematical formulation**:

$$\phi_\theta(h) = \text{shared-encoder}(h)$$
$$\phi_c(h) = W_c \cdot \phi_\theta(h), \quad \phi_s(h) = W_s \cdot \phi_\theta(h)$$

Loss:
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \alpha \|J_{\phi_c}(h_b)\|_F^2 + \beta \mathcal{L}_{\text{separation}}(\phi_s(h_b), \phi_s(h_m))$$

**Relation to AlphaSteer**: Maintains a single underlying encoder but allows geometric specialization, a middle ground between pure single-space and pure dual-space.

**Priority**: ★★☆☆☆

---

## 5. Recommended Primary Method: Locally-Gated Projector Network (LGP)

### Why LGP is the clear first choice

1. **Minimal departure from AlphaSteer**: Same mathematical primitives (SVD, pseudoinverse, linear algebra), same constraint philosophy, same plug-and-play deployment.
2. **Directly motivated by evidence**: P2 demands local projectors; LGP provides them. P4 shows global distortion; LGP eliminates it by construction.
3. **Single-space**: Operates entirely in the original activation space—no learned encoder/decoder, no second space.
4. **Trainable**: All components (anchors, local projectors, local steering directions, temperature) are computed from data.
5. **Interpretable**: Each anchor and its local projector have clear geometric meaning.
6. **Graceful degradation**: $K=1$ → AlphaSteer. Increasing $K$ → finer local adaptation. This gives us a natural hyperparameter sweep that includes AlphaSteer as the baseline.
7. **Narrative coherence**: "We upgraded AlphaSteer's global null-space to a locally-gated projector family"—this is one sentence, immediately understandable.

---

## 6. Mathematical Formulation of LGP

### 6.1 Setup

Let $h \in \mathbb{R}^d$ be an activation at layer $l$ of a transformer. We have:
- **Benign activations**: $\mathcal{H}_b = \{h_b^{(i)}\}_{i=1}^{N_b}$
- **Harmful activations**: $\mathcal{H}_m = \{h_m^{(i)}\}_{i=1}^{N_m}$
- **Refusal direction**: $r \in \mathbb{R}^d$

### 6.2 Step 1: Anchor Computation

Run k-means on $\mathcal{H}_b$ to obtain $K$ anchor centers:

$$\{z_k\}_{k=1}^K = \text{k-means}(\mathcal{H}_b, K)$$

### 6.3 Step 2: Soft Neighborhood Assignment

For any activation $h$, compute soft membership weights:

$$w_k(h) = \frac{\exp(-\|h - z_k\|^2 / \tau)}{\sum_{j=1}^K \exp(-\|h - z_j\|^2 / \tau)}$$

where $\tau > 0$ is a temperature parameter (can be set to the median squared inter-anchor distance, or learned).

### 6.4 Step 3: Local Null-Space Projectors

For each anchor $k$, collect its soft-weighted benign neighborhood:

$$H_b^{(k)} = \text{diag}(\sqrt{w_k(h_b^{(1)})}, \ldots, \sqrt{w_k(h_b^{(N_b)})}) \cdot \begin{bmatrix} h_b^{(1)\top} \\ \vdots \\ h_b^{(N_b)\top} \end{bmatrix}$$

Compute local null-space projector via SVD:

$$P_k = Q_k Q_k^\top, \quad Q_k = \text{null-space-basis}(H_b^{(k)}, \rho_k)$$

where $\rho_k$ is the local null-space ratio (can vary per anchor or be fixed).

### 6.5 Step 4: Local Steering Directions

For each anchor $k$, solve for the local steering direction using the harmful activations in its neighborhood:

$$\delta_k = \arg\min_\delta \|H_m^{(k)} P_k \delta - r\|^2 + \lambda \|P_k \delta\|^2$$

This has the same closed-form solution as AlphaSteer:

$$\delta_k = (X_k^\top X_k + \lambda P_k^\top P_k)^{-1} X_k^\top r$$

where $X_k = H_m^{(k)} P_k$.

### 6.6 Step 5: Position-Dependent Steering

The final steering at any point $h$ is:

$$s(h) = \lambda_{\text{strength}} \cdot \left(\sum_{k=1}^K w_k(h) \cdot S_k\right) h$$

where $S_k = P_k \delta_k$ is the local steering matrix at anchor $k$.

### 6.7 Efficient Implementation

Instead of storing $K$ full $d \times d$ matrices per layer, we can use a **low-rank factored form**:

$$S_k = Q_k (Q_k^\top \delta_k) = Q_k c_k$$

where $c_k = Q_k^\top \delta_k \in \mathbb{R}^{n_k}$ and $n_k = \text{dim(local null-space}_k\text{)}$. This reduces storage from $O(Kd^2)$ to $O(K \cdot d \cdot n_k)$.

The steering becomes:

$$s(h) = \lambda_{\text{strength}} \cdot \left(\sum_{k=1}^K w_k(h) \cdot Q_k c_k\right)^\top h$$

### 6.8 Key Properties

1. **Benign invariance**: For a benign activation $h_b$ near anchor $z_k$, $w_k(h_b) \approx 1$ and $P_k h_b \approx 0$ by construction, so $s(h_b) \approx 0$.
2. **Malicious steering**: For a harmful activation $h_m$ far from all benign anchors, the gating will weight anchors that allow non-zero projection, enabling refusal steering.
3. **Smoothness**: The softmax gating ensures $s(h)$ is smooth (infinitely differentiable), providing graceful transitions between neighborhoods.
4. **Reduction to AlphaSteer**: When $K=1$, $w_1(h) = 1$ for all $h$, and the formulation reduces exactly to the original AlphaSteer.

---

## 7. Experiment Plan

### 7.1 Minimum Viable Experiment (MVP)

**Goal**: Demonstrate that LGP with $K > 1$ outperforms AlphaSteer ($K = 1$) on the safety-utility tradeoff.

**Setup**:
- Model: Llama-3.1-8B-Instruct (same as AlphaSteer)
- Layers: Same steering layers as AlphaSteer ([8-19])
- Training data: Same benign/harmful/jailbreak splits
- $K \in \{1, 4, 8, 16, 32\}$ (K=1 is the AlphaSteer baseline)
- $\tau$: median squared distance among anchors (auto-calibrated)
- $\rho_k$: same null-space ratios as AlphaSteer per layer

**Evaluation metrics**:
- **Safety**: Jailbreak refusal rate on AdvBench, AIM, PAIR, GCG, AutoDAN, Cipher, ReneLLM
- **Utility**: AlpacaEval score, GSM8K accuracy, MATH accuracy, XSTest safe response rate
- **Diagnostic**: Per-anchor steering magnitude distribution on benign vs harmful

**Success criterion**: LGP with some $K > 1$ achieves ≥ AlphaSteer safety with ≥ AlphaSteer utility, OR achieves Pareto-dominant safety-utility tradeoff at some strength value.

### 7.2 Key Diagnostic Experiments

#### D1: Local Projector Quality
Measure the local benign residual: $\|P_k h_b\|$ for each anchor $k$ and each benign sample in its neighborhood.

- **Expected**: Local residuals are significantly smaller than global projector residuals.
- **Links to**: P2, P4

#### D2: Gating Behavior Analysis
Visualize the gating weights $w_k(h)$ for benign vs harmful activations.

- **Expected**: Benign activations have concentrated gating (high weight on one or few anchors); harmful activations may have more diffuse or atypical gating patterns.
- **Links to**: P1 (local incompatibility)

#### D3: Strength Sweep Comparison
Run LGP vs AlphaSteer across strength values $\{-0.1, -0.2, \ldots, -0.5\}$.

- **Expected**: LGP maintains higher utility at equivalent safety levels, or achieves higher safety at equivalent utility.
- **Links to**: P4 (global projector compromise)

#### D4: Refusal Direction Compatibility ($\mu(z)$ from P1)
For each anchor $k$, measure $\mu_k = \|P_k r\| / \|r\|$ (the fraction of the refusal direction preserved after local projection).

- **Expected**: $\mu_k$ varies across anchors but is generally higher than $\mu_{\text{global}} = \|\hat{P} r\| / \|r\|$.
- **Links to**: P1, P3

#### D5: One-Shot vs Multi-Step (touching P5)
Compare single-pass LGP steering against iterative re-computation of gating weights.

- **Protocol**: Apply $s(h)$, re-compute $h' = h + s(h)$, re-compute $s(h')$, iterate for $T$ steps with diminishing step size.
- **Expected**: Multi-step may yield marginal improvement but single-pass may suffice if local projectors are accurate.

### 7.3 Integration with Existing Experiments

| Existing Experiment | How LGP Connects |
|---|---|
| G1/G2 (local projector variability) | LGP's $\{P_k\}$ should recover the same variability patterns; higher $K$ should capture more |
| Phase 1.5A (global vs local distortion) | LGP with large $K$ should have near-zero local distortion |
| P6 experiments (space comparisons) | LGP operates in original space; if it works, it demonstrates that single-space suffices when constraints are localized |

---

## 8. When to Abandon Single-Space and Move to Dual-Space

### Clear failure indicators for single-space LGP

1. **Local projector incompatibility with refusal direction**: If $\mu_k = \|P_k r\| / \|r\| < \epsilon$ for a majority of anchors (say > 60%), then the refusal direction is systematically eliminated by local benign null-spaces. This would mean the "local incompatibility" (P1) is extreme and cannot be resolved by position-dependent gating alone.

   **Threshold**: If median $\mu_k < 0.1$ across anchors and layers, declare single-space insufficient.

2. **Pareto non-dominance at all $K$**: If no value of $K$ produces a safety-utility point that is Pareto-better than AlphaSteer ($K=1$), then localizing the projector does not help in the original space.

   **Threshold**: If LGP with $K \in \{4, 8, 16, 32\}$ fails to improve either safety or utility by ≥ 2% over AlphaSteer at any strength, declare single-space LGP insufficient.

3. **Benign charting vs separation anti-correlation**: If increasing $K$ (better local charting) systematically reduces separation quality (measured by malicious steering magnitude), then the two objectives are geometrically anti-correlated in the original space.

   **Threshold**: If Pearson correlation between "local projector quality" and "steering magnitude on harmful" is < -0.5 and statistically significant ($p < 0.01$), consider dual-space.

### Transition plan

If single-space LGP fails by the above criteria, proceed to:
1. **LCE (Candidate 2)**: Try a learned contractive space that may reconcile charting and separation.
2. If LCE also fails, **CSS (Candidate 4)**: Accept dual-space with explicit charting and separation spaces.

---

## 9. Risk Analysis and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| $K$ too small → behaves like AlphaSteer | Low (graceful degradation) | Sweep $K$ systematically |
| $K$ too large → overfitting, noisy local SVDs | Medium | Min neighbor count threshold; regularize local SVDs |
| Temperature $\tau$ mistuned → uniform gating or degenerate | Medium | Auto-calibrate from inter-anchor distances; validate on held-out set |
| Memory: $K \times d \times d$ per layer | Medium-High | Use low-rank factored form ($Q_k c_k$); quantize anchors |
| Local projectors too restrictive → kills refusal direction | High | Monitor $\mu_k$ diagnostic; relax null-space ratio locally |
| Inference latency: computing $K$ softmax weights per token | Low | $K$ is small (4-32); softmax over anchors is $O(Kd)$, negligible vs attention |

---

## 10. Shortest Implementation Roadmap

### Phase 1: LGP Core (1-2 weeks)
1. **Implement local steering utilities** (`src/utils/local_steering_utils.py`):
   - K-means anchoring on benign activations
   - Soft neighborhood assignment (kernel weights)
   - Local null-space projector computation (weighted SVD)
   - Local steering direction computation (regularized regression)
   - Low-rank factored storage

2. **Implement LocalAlphaSteer model** (`src/LocalAlphaSteerModel/`):
   - Locally-gated decoder layer (soft-blending of local steering matrices)
   - Model wrapper with local steering parameters

3. **Implement training pipeline** (`src/calc_local_steering.py`):
   - Load embeddings, cluster, compute local projectors, solve local regressions
   - Save local steering parameters (anchors, Q_k, c_k, τ)

### Phase 2: MVP Experiments (1-2 weeks)
4. Run LGP with $K \in \{1, 4, 8, 16, 32\}$ on Llama-3.1-8B
5. Evaluate on all existing benchmarks (AdvBench, AIM, PAIR, GCG, etc.)
6. Compute diagnostics D1-D4

### Phase 3: Analysis and Decision (1 week)
7. Compare LGP vs AlphaSteer across the Pareto frontier
8. If LGP succeeds: refine $K$, $\tau$, per-layer settings
9. If LGP fails: diagnose via failure criteria → proceed to LCE or CSS

### Phase 4: Paper-Ready (2-3 weeks)
10. Multi-model evaluation (Qwen-2.5, Gemma-2)
11. Ablation studies (K, τ, null-space ratio, low-rank vs full)
12. Multi-step experiments (touching P5)
13. Writeup

---

## Appendix A: Literature and Method Connections

### Methods closest to "single-space local nonlinear constrained steering"

| Method | What it solves | Gap to our need |
|--------|---------------|-----------------|
| **Conceptor-based Affine Steering** (NeurIPS 2025) | Compositional soft-projection for behavioral control | Uses fixed conceptors, not local neighborhoods |
| **SCANS** (AAAI 2025) | Safety-conscious layer-specific steering | No local projector adaptation |
| **Feature-Guided Activation Additions** (2025) | Sparse-autoencoder-guided steering | Feature selection, not local geometric constraint |
| **Contractive Autoencoders** (classic + modern) | Locally flat representations | Needs integration with projector framework |
| **Local Tangent Space Alignment** (manifold learning) | Preserves local geometry in dimensionality reduction | Unsupervised; needs adaptation for constrained steering |
| **Kernel Ridge Regression** (classic) | Position-dependent regression | Missing the projector constraint structure |
| **Mixture of Experts** (modern transformers) | Position-dependent routing | Missing null-space constraint philosophy |

### Key insight from literature synthesis

No existing method directly combines **local null-space projection** with **position-dependent gating** in the context of activation steering. The LGP proposal fills this gap by combining:
- AlphaSteer's constraint internalization philosophy
- K-means-based local neighborhood structure (from manifold learning)
- Soft kernel gating (from kernel methods / attention mechanisms)
- Local SVD-based projectors (from local PCA / tangent space methods)

This combination is novel in the steering literature.

### Methods that only solve benign charting

- PCA / Kernel-PCA: Good at local linearization but no steering mechanism
- Contractive Autoencoders: Flatten benign structure but require decode step
- Local Tangent Space Alignment: Preserves local geometry but not actionable for steering

### Methods that only solve malicious separation

- RFF (Random Fourier Features): Good at separating malicious from benign but poor at charting
- One-class SVM / isolation forests: Detection only, no steering
- Contrastive learning: Separation-focused, no constraint structure

### Methods closest to AlphaSteer's natural upgrade

- **LGP (this proposal)**: Closest—direct generalization of the mathematical structure.
- **Conceptor-based steering**: Similar philosophy (soft projection) but lacks local adaptation.
- **MoE-based steering**: Similar routing but lacks explicit null-space constraint.

---

## Appendix B: Relationship to Propositions P1-P6

| Proposition | Status | How LGP Addresses It |
|-------------|--------|---------------------|
| P1 (local incompatibility) | Untested | LGP's $\mu_k$ diagnostic directly measures local compatibility |
| P2 (local subspace family) | Strongly supported | LGP's $\{P_k\}$ is exactly this family |
| P3 (projection optimality) | Untested | LGP's $P_k \delta_k$ is the local projected steering; compare against unprojected |
| P4 (global projector compromise) | Partially supported | LGP eliminates global approximation error by construction |
| P5 (multi-step dynamics) | Untested | LGP naturally supports re-computation; test with iterative steering |
| P6 (representation space separability) | Refined | LGP tests whether single-space with local structure suffices |
