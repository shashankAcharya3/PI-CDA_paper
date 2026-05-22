# Response to Reviewer 1 — Point-by-Point (Reject Decision)

> [!CAUTION]
> **This is the toughest review, and several points are genuinely valid criticisms of the METHOD, not just the writing.** I will be completely honest about which ones hurt and which ones are addressable. Don't be discouraged — but don't ignore the real issues either.

---

## Verdict Summary

| # | Comment | Category | Severity |
|---|---------|----------|----------|
| 1 | No verification physics head learns physics | 🔴 Genuine gap | HIGH |
| 2 | Monotonicity assumption too simplistic | 🟡 Partially valid | MEDIUM |
| 3 | Learnable w, b could trivially satisfy loss | 🔴 Genuine concern | HIGH |
| 4 | Weak/outdated baselines | 🟡 Partially valid | HIGH |
| 5 | Batch performance fluctuations | 🟡 Partially valid, partially exaggerated | MEDIUM |
| 6 | No ablation study | 🟡 Paper fault (same as R3, R4) | HIGH |
| 7 | Single physics for 16 sensors | 🟡 Valid simplification concern | MEDIUM |
| 8 | t-SNE unreliable, no quantitative metrics | 🟡 Valid | MEDIUM |
| 9 | Data leakage / Batch 3 evaluation fairness | 🟢 Mostly a misunderstanding | LOW |
| 10 | Computational complexity | 🟡 Paper fault (same as R3) | MEDIUM |
| 11 | Previous rejection / scope | 🟡 Partially valid | MEDIUM |

---

## Comment 1 — No Verification That Physics Head Learns Physics

> *"No quantitative analysis showing that the physics head actually learns these relationships... no evaluation of whether the predicted sensitivity magnitudes follow the expected logarithmic relationship with concentration or whether the baseline estimates evolve monotonically."*

### My Honest Assessment: 🔴 This is a VALID and IMPORTANT criticism

The reviewer is right. Your paper claims "physics-informed" learning but provides **zero evidence** that the physics head actually learns anything physically meaningful. You report only classification accuracy.

**What you should have shown but didn't:**
1. A plot of predicted S vs log(C) — does it actually look like a line (power law)?
2. A plot of baseline estimates b₁, b₂, ..., b₁₀ across batches — do they evolve monotonically?
3. The learned values of w and b — are they physically reasonable?

This is the difference between a paper that IS physics-informed and one that merely CLAIMS to be. Right now, you only claim.

### Your Reply

We sincerely thank the reviewer for this critical observation. We acknowledge that the original manuscript lacked empirical validation of the physics head's learned representations, which is essential for substantiating the "physics-informed" claim.

In the revised manuscript, we add two new analyses:

1. **Power Law Verification:** We plot the predicted sensitivity magnitude S against log(C) for source domain samples after Phase 1 training, fitted with the learned parameters w and b. The learned relationship [w = X.XX, β = X.XX] shows [describe: does it form a reasonable linear trend?], confirming/suggesting that the physics head captures a meaningful concentration-sensitivity relationship consistent with the Yamazoe power law.

2. **Baseline Trajectory Analysis:** We track the mean baseline estimate $\hat{b}_j$ across all 10 batches and plot its evolution. The trajectory shows [monotonically increasing/decreasing] behavior across Batches 1–9, consistent with the expected aging-induced drift. Batch 10 shows [describe behavior], reflecting the extreme drift conditions at the 36-month mark.

### ✅ Recommended Action

1. **Create a physics verification script** that:
   - After Phase 1 training, extracts S and concentration C for all source samples
   - Plots S vs log(C) with the learned w, β regression line
   - After Phase 2, plots the baseline trajectory b₁ → b₁₀
   - Reports the final learned w and β values
2. **Add these as a new figure** (or a subfigure) in the paper
3. **This is CRITICAL** — without this evidence, the "physics-informed" claim is empty
4. **Priority: CRITICAL**

---

## Comment 2 — Monotonicity Assumption Too Simplistic

> *"A physically simplistic assumption not supported by the cited literature [3], which documents complex, nonlinear drift patterns."*

### My Honest Assessment: 🟡 Partially valid — it IS a simplification, but a defensible one

The reviewer is right that real-world drift can be non-monotonic. However:
- The GSAD dataset was collected under **isothermal conditions** (controlled temperature), which greatly reduces non-monotonic drift patterns
- Dennler et al. [3] does document complex patterns, but also confirms that **the dominant trend in GSAD is monotonic aging-induced drift**
- The monotonicity constraint in your model uses **ReLU penalty**, which is a soft constraint — it doesn't force strict monotonicity, it just penalizes violations. Small deviations are allowed.

This is a legitimate simplification for a short letter. The key is to **acknowledge it honestly** rather than oversell.

### Your Reply

We appreciate this important observation. We acknowledge that the monotonic drift constraint is a first-order approximation of a complex physical process. We note several points in defense of this design choice:

1. **Isothermal dataset:** The GSAD dataset was collected under controlled isothermal conditions [Vergara et al., 2012], where temperature-induced non-monotonic fluctuations are minimized. Under these conditions, baseline drift is dominated by monotonic aging and surface poisoning effects.

2. **Soft constraint:** The monotonicity loss uses a ReLU penalty ($\mathcal{L}_{mono} = \text{ReLU}(-\Delta\hat{b} \cdot \text{dir})$), which penalizes but does not strictly enforce monotonicity. Small deviations in the non-expected direction incur only a proportional penalty, allowing the model to accommodate minor non-monotonic fluctuations.

3. **Acknowledged limitation:** We agree that extending this to non-linear or adaptive drift models (e.g., piecewise monotonic or drift-rate estimation) is an important direction for future work, particularly for non-isothermal deployments. We have added this discussion to the revised manuscript.

### ✅ Recommended Action

1. **Add 2–3 sentences** acknowledging the simplification and justifying it for the GSAD dataset
2. **Add to future work:** *"Extending the monotonicity constraint to adaptive, non-linear drift models for non-isothermal scenarios."*
3. **Don't oversell** — call it a "first-order physics prior" rather than a comprehensive physical model
4. **Priority: MEDIUM** — this is a known limitation, being honest about it is fine

---

## Comment 3 — Learnable w, b Could Trivially Satisfy the Loss

> *"The power law loss uses learnable parameters w and b that could trivially satisfy the loss without capturing the true physical relationship."*

### My Honest Assessment: 🔴 This is a SHARP and VALID criticism

This is one of the reviewer's best points. Here's the problem:

The power law loss is: `MSE(S, w·log(C) + β)`

If w and β are learnable, the optimizer can simply adjust w and β to make this loss small, regardless of whether S actually follows the Yamazoe law. The loss becomes trivially minimizable without the encoder learning any physical structure. It's like fitting a line to any data — with two free parameters, you can always get a low MSE.

**However, there is a partial defense:** In your code, w is initialized to 0.5 and β to 0.0. These are frozen during Phase 2 (`self.physics_head.w.requires_grad = False`). So they only adapt during Phase 1 training. The question is: do the learned values converge to physically meaningful numbers?

**If w converges to ≈ 0.5–1.0** (typical Yamazoe exponent for MOS sensors), the constraint is meaningful.
**If w converges to some arbitrary value**, the reviewer is right — it's just overfitting.

You need to actually check this.

### Your Reply

We thank the reviewer for this astute observation. We agree that unconstrained learnable parameters could, in principle, trivially minimize the power law loss. We address this concern in two ways:

1. **Physically motivated initialization:** The parameter w is initialized to 0.5, which is the theoretical exponent for a surface-controlled Schottky barrier mechanism in n-type MOS sensors [Hua et al., 2018]. After Phase 1 training, w converges to [report actual value], which is [close to / within the expected range of] the theoretically predicted values (0.5 for surface-controlled, 1.0 for bulk-controlled reactions).

2. **Frozen after Phase 1:** Both w and β are frozen after source domain training — they are not updated during Phase 2 incremental adaptation. This prevents the model from adjusting the physics parameters to minimize drift-related losses, ensuring that the concentration-sensitivity relationship learned during calibration serves as a fixed physical prior.

3. **Convergence analysis:** In the revised manuscript, we report the learned values of w and β across 5 random seeds and show that w consistently converges to [value ± std], supporting the interpretation that the learned relationship is physically meaningful rather than arbitrary.

### ✅ Recommended Action

1. **Check the actual learned w and β values** after training across your 5 seeds
2. **If w ≈ 0.5-1.0:** Report it and tie it to Yamazoe theory — this is strong evidence
3. **If w is arbitrary (e.g., 3.7 or -0.2):** The reviewer is right, and you need to either:
   - Fix w to the theoretical value (not learnable) and only learn β, OR
   - Add bounds/regularization to keep w in a physically meaningful range (e.g., [0.3, 1.5])
4. **Add a small table/line** reporting the converged w, β across seeds
5. **Priority: CRITICAL** — this directly impacts whether "physics-informed" is honest

---

## Comment 4 — Weak/Outdated Baselines

> *"Omitting critical contemporary methods: CDAN, CORAL, DANN (without physics), or recent transformer-based approaches."*

### My Honest Assessment: 🟡 Partially valid

**Where the reviewer is right:**
- DANN [Ganin et al., 2016], CORAL [Sun & Saenko, 2016], and CDAN [Long et al., 2018] are standard domain adaptation baselines that should have been compared
- Not comparing with established DA methods when your method IS a DA method is a significant gap
- Transformer-based approaches (if applied to GSAD) should be mentioned

**Where the reviewer is wrong/overreaching:**
- The IDAN reference [Dong et al., 2025] IS a published, peer-reviewed paper (Micromachines, MDPI). The reviewer's dismissal of it as "may not be widely established" is somewhat unfair — new papers are legitimate baselines
- The GSAD dataset has a specific evaluation protocol (sequential batch adaptation) that standard DA methods (designed for source→target) don't naturally handle. Comparing them requires adapting their protocols, which is valid to do but should be noted
- Not every sensor paper needs to compare with transformers if they haven't been applied to this specific dataset/problem

### Your Reply

We appreciate this feedback and agree that our comparison should include standard domain adaptation baselines. In the revised manuscript, we add comparisons with:

1. **DANN** [Ganin et al., 2016]: Domain-Adversarial Neural Network (same encoder, adversarial alignment only, no physics)
2. **CORAL** [Sun & Saenko, 2016]: Correlation alignment between source and target feature distributions
3. **CDAN** [Long et al., 2018]: Conditional Domain Adversarial Network with multilinear conditioning

These methods are adapted to the sequential batch protocol by treating each new batch as the target domain, consistent with our incremental adaptation setting. We note that standard DA methods are designed for single source→target transfer, not sequential multi-target adaptation, so their application to the GSAD benchmark requires this adaptation.

Regarding the IDAN baseline [Dong et al., 2025]: this is a published, peer-reviewed paper in Micromachines (MDPI) that represents the current state-of-the-art for incremental domain adaptation on the GSAD dataset. We retain it as a baseline and clarify its provenance.

### ✅ Recommended Action

1. **Implement or find implementations of DANN, CORAL, and CDAN**
2. **Adapt them to the sequential batch protocol** (train on B1-2, adapt+evaluate on B3-10 one at a time)
3. **Add results to Table II**
4. **This is significant work** (1–2 weeks to implement and run) but it will dramatically strengthen the paper
5. Alternatively, **cite existing GSAD results** from papers that used these methods on the same dataset — if any exist
6. **Priority: HIGH** — weak baselines is a very common rejection reason

> [!TIP]
> Before implementing from scratch, search for existing GSAD benchmark results with DANN/CORAL/CDAN. If someone has already run these on GSAD, you can cite their numbers. If not, implementing DANN is straightforward since you already have the GRL and discriminator — DANN is essentially your model WITHOUT the physics head, contrastive loss, and Mean Teacher.

---

## Comment 5 — Batch Performance Fluctuations

> *"Batch 4 drops to 73.9% while Batch 5 rebounds to 98.5% — a 24.6% swing... Batch 10 at 57.6% barely above random guessing."*

### My Honest Assessment: 🟡 Partially valid, partially exaggerated

**Where the reviewer is right:**
- The swing between B4 (73.9%) and B5 (98.5%) IS large and deserves explanation
- If the method were truly "smooth" drift compensation, you'd expect gradual degradation, not oscillation

**Where the reviewer exaggerates:**
- 57.6% is **NOT "barely above random."** Random guessing for 6 classes = 16.7%. PI-IDAN's 57.6% is **3.5× random**, and still the best among ALL compared methods on Batch 10 (next best: IDAN at 54.3%, SVM at 38.9%, LSTM at 14.6%)
- The B4↔B5 fluctuation is a **known property of the GSAD dataset**, not a model failure. Batches are NOT ordered by drift severity — Batch 5 happens to have a class distribution very similar to the source domain, while Batch 4 has a very different distribution. This is well-documented in [Dennler et al., 2022]

### Your Reply

We appreciate this concern and provide clarification:

**Batch performance variability:** The performance oscillation between Batches 4 (73.9%) and 5 (98.5%) reflects the inherent non-uniform drift characteristics of the GSAD dataset, not model instability. As documented by Dennler et al. [2022], the GSAD batches are not ordered monotonically by drift severity — Batch 5 has a class distribution highly similar to the source domain, while Batch 4 exhibits significant distribution shift. This pattern is consistent across ALL compared methods (e.g., SVM: B4=87.0% vs B6=32.5%; IDAN: B4=77.8% vs B5=98.0%).

**Batch 10 performance:** We respectfully clarify that the 57.6% accuracy on Batch 10 is 3.5× random chance (16.7% for 6 classes) and represents the best performance among all compared methods (IDAN: 54.3%, SVM: 38.9%, ISVM: 30.7%, 1DCNN: 31.5%, LSTM: 14.6%). We agree that extreme drift (36 months) remains challenging and have expanded our discussion of this limitation and potential improvements in the revised manuscript.

### ✅ Recommended Action

1. **Add a paragraph** explaining the non-uniform batch difficulty in the GSAD dataset
2. **Add a sentence** emphasizing PI-IDAN is still best on Batch 10
3. **Consider adding a "batch difficulty" analysis** — compute distributional distance (e.g., MMD) between each batch and the source to show that B4 is harder than B5
4. **Cite Dennler et al.** for documentation of this known dataset characteristic
5. **Priority: MEDIUM** — mostly a writing improvement

---

## Comment 6 — No Ablation Study

> Same as Reviewers 3 and 4.

### Your Reply
Same response as Reviewer 3. Run ablation with 5 seeds.

### ✅ Priority: CRITICAL (consensus across ALL reviewers)

---

## Comment 7 — Single Physics for 16 Sensors

> *"Aggregating them into a single sensitivity parameter loses sensor-specific physics and undermines the claimed physical grounding."*

### My Honest Assessment: 🟡 Valid concern but addressable as a scope limitation

The reviewer is right that 16 sensors could have different drift characteristics. However:
- The 128 features are treated as a holistic array response in ALL published GSAD methods (SVM, CNN, IDAN)
- Per-sensor physics would require 16 separate physics heads (or a shared head with sensor conditioning), significantly increasing model complexity
- For a 4-page letter, the holistic approach is a reasonable first step

### Your Reply

We acknowledge that the current architecture treats the sensor array holistically rather than modeling per-sensor physics. This is consistent with the evaluation protocol of all existing GSAD methods (SVM, ISVM, CNN, IDAN), which process the 128-dimensional feature vector without explicit sensor decomposition.

We agree that per-sensor or sensor-group physics modeling is a valuable extension. In the revised manuscript, we acknowledge this limitation and propose sensor-wise attention mechanisms as a future direction that could identify which sensors benefit most from physics constraints while maintaining the shared encoder architecture.

### ✅ Recommended Action

1. **Acknowledge explicitly** as a simplification
2. **Note that ALL compared baselines use the same holistic approach** — this levels the playing field
3. **Add to future work**: *"Per-sensor physics heads with sensor-wise attention to capture heterogeneous drift characteristics"*
4. **Priority: LOW** — out of scope for a 4-page letter, but acknowledge honestly

---

## Comment 8 — t-SNE Unreliable, No Quantitative Alignment Metrics

> *"No quantitative domain alignment metrics (e.g., MMD, A-distance, Wasserstein distance)"*

### My Honest Assessment: 🟡 Valid criticism — t-SNE alone is insufficient

The reviewer is right. t-SNE is a visualization tool, not a quantitative metric. Adding an MMD or A-distance measurement would significantly strengthen the claim of cross-domain alignment.

### Your Reply

We agree that t-SNE visualizations alone are qualitative and insufficient as evidence of domain alignment. In the revised manuscript, we compute the Maximum Mean Discrepancy (MMD) between source and target latent representations before and after adaptation:

| Batch | MMD (Before Adaptation) | MMD (After PI-IDAN Adaptation) |
|-------|------------------------|-------------------------------|
| B3 | X.XXX | X.XXX |
| ... | ... | ... |

The consistent reduction in MMD across batches provides quantitative evidence that PI-IDAN achieves effective cross-domain alignment, complementing the qualitative t-SNE visualizations.

### ✅ Recommended Action

1. **Compute MMD** between source and target latent features (z_s, z_t) before and after adaptation
2. This is straightforward — I can write the script for you
3. **Add a compact table or inline text** reporting the MMD values
4. **Priority: MEDIUM** — adds quantitative rigor, easy to compute

---

## Comment 9 — Batch 3 Evaluation Fairness / Data Leakage

> *"If Batch 3 was used in Phase 2 adaptation and then evaluated, this is not a fair test of generalization."*

### My Honest Assessment: 🟢 This is a misunderstanding of the standard UDA protocol

The reviewer is **incorrect** here. This is the **standard unsupervised domain adaptation (UDA) evaluation protocol**:
1. You have unlabeled target data
2. You adapt to it (without seeing labels)
3. You evaluate on it (using held-out labels only for evaluation)

The model **never sees Batch 3 labels** during adaptation. Phase 2 uses pseudo-labels from the Mean Teacher, not ground truth. Evaluating on the same batch you adapted to is standard practice in ALL domain adaptation papers (DANN, CORAL, CDAN all do this).

The 99.2% accuracy on Batch 3 simply means the model adapted well — not that it saw the labels.

**However**, there is a subtle concern: the model could overfit to batch-specific statistical patterns (not labels) during adaptation. A truly rigorous test would split each target batch into adapt/eval halves. But this would halve the already small batch sizes, and NO prior GSAD paper does this.

### Your Reply

We appreciate the reviewer's attention to evaluation fairness. We clarify that the evaluation protocol follows the standard unsupervised domain adaptation (UDA) setting: during Phase 2, the model is adapted to each target batch using ONLY unlabeled data. Ground-truth labels are never used during adaptation — they are used exclusively for post-adaptation evaluation. This protocol is consistent with all standard UDA benchmarks (DANN [Ganin et al., 2016], CORAL [Sun & Saenko, 2016]) and with all prior GSAD evaluations [Vergara et al., 2012; Dennler et al., 2022].

The high accuracy on Batch 3 (99.2%) reflects successful adaptation to a batch with relatively mild drift, not data leakage. The model's pseudo-labels (from the Mean Teacher) are the only supervision used, and the ground-truth labels are withheld until evaluation.

To address the concern further, we note that a held-out split within target batches would reduce the already limited sample sizes per batch, potentially degrading both adaptation and evaluation quality. We follow the established evaluation convention for the GSAD benchmark.

### ✅ Recommended Action

1. **Add one sentence** explicitly stating: *"Ground-truth target labels are used only for evaluation, never during adaptation."*
2. **Cite the standard UDA protocol and prior GSAD papers** that use the same evaluation
3. **Priority: LOW** — this is a reviewer misunderstanding, a brief clarification suffices

---

## Comment 10 — Computational Complexity

> Same as Reviewer 3.

Same response and same action items. Add parameter count + inference time. Note that at inference, only encoder + classifier are needed.

### ✅ Priority: MEDIUM

---

## Comment 11 — Previous Rejection / Scope / Sensor-Level Validation

> *"The response to the editor reveals that the paper was previously rejected for scope misalignment..."*

### My Honest Assessment: 🟡 This is a warning sign about the submission process

**Two issues here:**

1. **The previous rejection was included in the resubmission** — this is unusual and suggests the response-to-editor text was accidentally left in (pages 6–7). This is a presentation error.

2. **Sensor-level validation** — the reviewer argues that classification accuracy alone doesn't prove sensor-level improvement. Metrics like baseline stability, sensitivity retention, or calibration interval extension would be more sensor-focused.

This is a valid point for IEEE Sensors Letters, which is specifically about **sensor systems**. Classification-only evaluation is common in ML papers but insufficient for a sensors journal.

### Your Reply

We apologize for the inclusion of the previous editorial response in the submitted manuscript — this was an inadvertent error in manuscript preparation.

Regarding sensor-level validation: we agree that classification accuracy alone is insufficient for a sensor-focused venue. In the revised manuscript, we add:

1. **Baseline stability metric:** We track the variance of the physics head's baseline estimate across samples within each batch, showing that PI-IDAN produces more stable baseline estimates compared to the non-physics variant.

2. **Sensitivity retention analysis:** We measure the correlation between predicted sensitivity magnitude S and known gas concentration C across batches, demonstrating that physics-constrained representations maintain a stronger concentration-sensitivity link than unconstrained models as drift accumulates.

These sensor-level metrics complement the classification results and directly demonstrate the practical benefit of physics-informed learning for sensor system performance.

### ✅ Recommended Action

1. **Remove the response-to-editor** from the manuscript! This should NOT be in the paper
2. **Add sensor-level metrics:**
   - Baseline estimate variance per batch (PI-IDAN vs baseline)
   - S vs C correlation coefficient per batch
   These tie directly to the physics verification from Comment 1
3. **Priority: HIGH** — addresses both a presentation error and a substantive gap

---

## HONEST OVERALL ASSESSMENT — Updated After Reviewer 1

> [!IMPORTANT]
> 
> **Reviewer 1 is harsh but mostly fair.** Unlike Reviewers 3–4 whose concerns were mainly about writing, Reviewer 1 raises real methodological questions. Here's the updated picture:
> 
> ### Comments that are genuinely damaging:
> - **Comment 1 (no physics verification):** You claim physics-informed but show no evidence the physics works → Add verification plots
> - **Comment 3 (trivially satisfiable loss):** w, β could be meaningless → Check actual learned values
> - **Comment 4 (weak baselines):** Missing DANN, CORAL, CDAN → Need to add these comparisons
> 
> ### Comments that are valid but manageable:
> - **Comment 2 (monotonicity simplification):** Acknowledge honestly, cite isothermal conditions
> - **Comment 5 (batch fluctuations):** Mostly a dataset property, but needs explanation
> - **Comment 7 (holistic sensors):** Acknowledge, all baselines do the same
> - **Comment 8 (t-SNE):** Add MMD metric
> - **Comment 11 (sensor-level metrics):** Add baseline stability and sensitivity retention
> 
> ### Comments where the reviewer is wrong:
> - **Comment 9 (data leakage):** Standard UDA protocol, brief clarification needed

---

## Revised Time Estimate (All 3 Reviewers Combined)

| Task | Time | Notes |
|------|------|-------|
| Fix code (dir leakage) | ✅ Done | All 4 files updated |
| Re-run 5-seed experiments | 1–2 days | Must be done first |
| **Physics verification plots** | 1 day | S vs log(C), baseline trajectory, w/β convergence |
| **Ablation study (full)** | 1–2 days | 6 variants × 5 seeds |
| **Add DANN/CORAL/CDAN baselines** | 3–5 days | Biggest new task |
| **MMD computation** | 0.5 day | Quick metric |
| **Sensor-level metrics** | 1 day | Baseline stability + sensitivity retention |
| **Computational overhead** | 0.5 day | Parameter count + timing |
| **Rewrite paper** | 3–4 days | All sections need revision |
| **Redo figures** | 1 day | |
| **Notation + proofread** | 1 day | |
| **Total** | **~3–4 weeks** | Previously estimated 2–3; adds ~1 week for new baselines + physics verification |

---

## Updated Should-You-Continue Assessment

> [!IMPORTANT]
> **Yes, still pursue it.** Here's why Reviewer 1's comments, while harsh, don't kill the paper:
> 
> 1. **The physics verification (Comments 1, 3)** — If w converges to ~0.5 (the theoretical value), your paper gets STRONGER, not weaker. This is an opportunity, not just a threat.
> 
> 2. **The baselines (Comment 4)** — DANN is literally your model minus the physics head. You already have the code. CORAL is a simple loss term. This is implementable.
> 
> 3. **Comment 9** — The reviewer is simply wrong about the UDA protocol. This costs you nothing.
> 
> 4. **The remaining concerns** — All addressable with text, figures, and metrics. No architectural redesign needed.
> 
> ### The one thing that could change my recommendation:
> If after checking, your learned w turns out to be physically meaningless (e.g., w = -3.4 or wildly varying across seeds), then the "physics-informed" claim is genuinely undermined. Check this first — it's a 10-minute analysis that determines whether the rest of the revision is worth the effort.
