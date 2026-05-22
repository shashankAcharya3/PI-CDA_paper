# Response to Reviewer 3 — Point-by-Point

> [!NOTE]
> This document contains: (1) **Your reply** to copy into the rebuttal, and (2) **Recommended action** — what you actually need to do in the revised manuscript. Replies are written in a professional, grateful tone since this reviewer was supportive.

---

## 2.1 Experimental Details and Reproducibility

---

### Comment 2.1.1 — Hyperparameter Selection Justification

> *"The manuscript empirically sets hyperparameter values (e.g., λ_cont=1.0, λ_power1=0.1, λ_mono2=0.5, learning rates for Phase 1/2) but provides no justification for these choices..."*

#### Your Reply

We thank the reviewer for this important observation. We acknowledge that the original manuscript lacked justification for the hyperparameter choices. In the revised manuscript, we provide a systematic justification grounded in the design rationale of the two-phase training scheme:

- **Phase 1 physics weights (λ_power1=0.1, λ_mono1=0.1):** In Phase 1, the primary objective is to learn discriminative gas features from labeled source data. The physics losses serve as soft regularizers at this stage rather than dominant constraints, hence the lower weighting. Setting them too high during initial training risks suppressing the task loss signal before the encoder learns meaningful representations.

- **Phase 2 physics weights (λ_mono2=0.5):** During incremental adaptation, the physics constraints become more critical because the model must resist overfitting to drift-corrupted pseudo-labels. The increased monotonicity weight enforces stronger physical consistency, preventing the baseline estimate from fluctuating unrealistically during adaptation.

- **Classifier learning rate (10× slower in Phase 2):** This follows established continual learning practices where the classification head is updated conservatively to preserve source domain knowledge, while the encoder adapts more aggressively to align new domain representations. This strategy is analogous to discriminative fine-tuning used in transfer learning [Howard & Ruder, 2018].

Additionally, we have included a new ablation study (Table X in the revised manuscript) that quantifies the contribution of each physics component, providing empirical evidence for these design decisions.

#### ✅ Recommended Action

1. **Add 2–3 sentences** in Section II explaining the rationale (copy from above)
2. **Run the ablation study** (`run_ablation_study.py`) with 5 seeds and include the ablation table
3. Optionally, add a brief sensitivity analysis showing performance at λ_mono2 ∈ {0.1, 0.3, 0.5, 0.7, 1.0} — even a single sentence like *"Performance was robust across λ_mono2 ∈ [0.3, 0.7], with 0.5 yielding the best trade-off"* would help
4. **Priority: HIGH** — this directly affects perceived scientific rigor

---

### Comment 2.1.2 — Dataset Preprocessing Details

> *"How are the 128-dimensional features derived from 16 MOS sensors? Are the sensor responses normalized to a specific range before Z-score? Are outliers or missing values present?"*

#### Your Reply

We appreciate the reviewer's attention to reproducibility. The GSAD dataset [Vergara et al., 2012] provides 128 features derived from 16 metal-oxide semiconductor gas sensors, where each sensor contributes 8 features: two steady-state resistance values (maximum and minimum resistance during the measurement cycle) and six transient response characteristics (exponential moving averages at three time constants for both the rise and decay phases). This structure is defined by the original dataset authors and is used without modification.

Regarding preprocessing: Z-score normalization (zero mean, unit variance) is computed exclusively from Batch 1 data and applied uniformly to all subsequent batches. No additional range normalization (e.g., [0,1] scaling) is applied prior to Z-scoring, and no outlier removal or imputation is performed. This is intentional — the drift-induced distributional shift is the phenomenon we aim to model, and any batch-wise normalization would destroy this signal. The original GSAD dataset documentation confirms no missing values are present.

We have added these details to a new "Experimental Setup" subsection in the revised manuscript.

#### ✅ Recommended Action

1. **Add an "Experimental Setup" subsection** at the start of Section III with the text above
2. Verify the exact 8 features per sensor from the original Vergara et al. paper (the 8 features are: max conductance, max conductance / baseline, area under curve, max-min difference, and similar derived features — double-check against the UCI page)
3. **Priority: MEDIUM** — standard reproducibility detail

---

### Comment 2.1.3 — Training Implementation Details

> *"Missing: batch size, hardware/software, number of random seeds..."*

#### Your Reply

We thank the reviewer for highlighting these missing details. In the revised manuscript, we specify the full training configuration:

- **Framework:** PyTorch 2.x
- **Hardware:** [INSERT YOUR ACTUAL HARDWARE — e.g., NVIDIA RTX 3060 GPU / Apple M1 / CPU-only]
- **Batch size:** 64 for both Phase 1 and Phase 2
- **Phase 1 training:** 50 epochs with Adam optimizer (lr=0.001) and cosine annealing scheduler
- **Phase 2 adaptation:** Adaptive epoch count per batch (5–40 epochs based on pre-adaptation accuracy), Adam optimizer (lr=0.0001), gradient clipping (max norm 1.0)
- **Random seeds:** All results are reported as mean ± standard deviation across 5 independent runs with seeds {42, 123, 456, 789, 1024}
- **Reproducibility:** Seeds are set for Python, NumPy, PyTorch, and CUDA with `torch.backends.cudnn.deterministic = True`

#### ✅ Recommended Action

1. **Add a paragraph** with the above details to the "Experimental Setup" subsection
2. **Fill in your actual hardware** — check what you ran this on
3. **Critical: Fix the results discrepancy.** Your 5-seed mean is 80.2±4.1%, not 82.54±1.9% as currently reported. You MUST either:
   - **(a)** Re-run all experiments cleanly and report the true 5-seed numbers, OR
   - **(b)** Report the single-seed result clearly as *"best single run"* alongside the 5-seed mean
   - I **strongly recommend option (a)** — honesty here prevents future problems
4. **Priority: CRITICAL** — the numbers must be accurate

---

### Comment 2.1.4 — Memory Replay Buffer Design

> *"The manuscript does not specify: buffer size, sample selection method..."*

#### Your Reply

We appreciate this observation and have clarified the memory replay buffer design in the revised manuscript. The buffer consists of two components:

1. **Core Memory:** All labeled samples from source Batches 1 and 2 are retained permanently, serving as ground-truth anchors throughout the incremental adaptation process.

2. **Distilled Memory:** After adapting to each target batch, high-confidence pseudo-labeled samples are added to the buffer. Sample selection uses an ensemble agreement criterion: only samples where both the neural network classifier and a Random Forest (trained on the encoder's latent space) agree on the predicted class, and the neural network confidence exceeds an adaptive class-wise threshold (median confidence per class, minimum 0.4), are retained. A maximum of 50 samples per gas class per batch is enforced to prevent buffer bloat while maintaining class balance. During training, the combined buffer is sampled using inverse-frequency class weighting to ensure equitable representation of all gas classes.

This design balances storage efficiency with catastrophic forgetting prevention by maintaining a compact yet representative memory of all encountered distributions.

#### ✅ Recommended Action

1. **Add the above description** to Section II.B (Phase 2), condensed to fit the 4-page limit
2. A compact version for the letter format:
   > *"The replay buffer retains all source samples (Batches 1–2) and up to 50 high-confidence pseudo-labeled samples per class per adapted batch, selected via ensemble agreement between the neural classifier and a Random Forest. Class-balanced sampling via inverse-frequency weighting ensures equitable replay."*
3. **Priority: HIGH** — this is a core design component

---

## 2.2 Technical Clarification and Notation

---

### Comment 2.2.1 — Notation Inconsistencies

> *"λ_proso instead of λ_proto, typos in pseudo-label notation, usel instead of L_Phase1..."*

#### Your Reply

We sincerely apologize for these notation errors. They are typographical mistakes that occurred during manuscript preparation and do not reflect any ambiguity in the underlying method. All notation inconsistencies have been thoroughly corrected in the revised manuscript:

- `λ_proso` → `λ_proto` (prototype loss weight)
- `ŷ_j^{fhigh-cord}` → `ŷ_j^{t,\text{high-conf}}` (high-confidence pseudo-labels for target batch j)
- `usel` → `\mathcal{L}_{\text{Phase1}}` (total Phase 1 loss)

We have conducted a complete notation audit of the revised manuscript to ensure consistency throughout.

#### ✅ Recommended Action

1. **Do a complete line-by-line notation audit** of the entire LaTeX source
2. Create a notation table (even if just for your own reference) listing every symbol and its meaning
3. Have each co-author independently proofread the notation
4. Check that every symbol defined in text matches the equations exactly
5. **Priority: HIGH** — sloppy notation undermines reviewer confidence in the work

---

### Comment 2.2.2 — Physics Head Design Clarification

> *"How does the 2D MLP output map to S and b? How is baseline estimated from z when the input contains gas-present responses?"*

#### Your Reply

We thank the reviewer for this insightful question. We clarify the physics head design:

The sensitivity magnitude S and baseline estimate b are obtained through different computation paths:

- **Sensitivity magnitude S:** S is computed as the L2 norm of the latent vector z, i.e., S = ||z||₂. This captures the overall activation magnitude of the encoded sensor response. Under the Yamazoe power law, a sensor's sensitivity is related logarithmically to gas concentration. By constraining ||z||₂ to follow this relationship, we ensure the representation's magnitude encodes physically meaningful sensitivity information.

- **Baseline estimate b:** b is the second output neuron of the physics MLP (64→32→2). The MLP learns to estimate the sensor's resting-state baseline from the encoded features. Although the input includes gas-present responses, the encoder is trained to separate gas-specific from drift-specific information in the latent space. The physics head extracts the drift-related component (baseline) from z, analogous to how a denoising autoencoder separates signal from noise. The monotonicity loss then ensures this estimated baseline evolves monotonically, consistent with the expected aging-induced drift trajectory.

We have revised Section II.A to include this clarification.

#### ✅ Recommended Action

1. **Rewrite the physics head paragraph** in Section II to clearly state:
   - S = ||z||₂ (not from MLP)
   - b = second output of MLP
2. Add a brief physical intuition sentence
3. **Fix the architecture diagram** if it currently shows both S and b coming from the MLP
4. **Priority: HIGH** — this is a core technical clarification

---

### Comment 2.2.3 — Monotonic Drift Direction (dir)

> *"How is dir determined from the initial source batches? Is it fixed or adaptive per sensor?"*

#### Your Reply

The drift direction parameter dir ∈ {+1, −1} is determined empirically by comparing the mean sensor response between the earliest (Batch 1) and latest (Batch 10) batches. Specifically, dir = sign(mean(feat₀|Batch 10) − mean(feat₀|Batch 1)), where feat₀ denotes the first sensor feature. If the mean response increases over time, dir = +1; otherwise, dir = −1.

This direction is computed once using the first sensor channel and is applied globally across all sensors. This simplification is justified for MOS gas sensor arrays because baseline drift due to aging and surface poisoning is predominantly monotonic and unidirectional for a given sensor type under isothermal operation [Dennler et al., 2022; Vergara et al., 2012]. Extending this to per-sensor adaptive directions is a promising avenue for future work, particularly for heterogeneous sensor arrays where different sensor materials may exhibit opposing drift trends.

We have added this explanation to Section II.A in the revised manuscript.

#### ✅ Recommended Action

1. **Add 2 sentences** to Section II explaining how dir is computed
2. Acknowledge the single-direction limitation in the discussion/conclusion
3. Consider whether computing dir from Batch 1 vs Batch 2 (your actual source batches) would be more principled than using Batch 10 (which you wouldn't have access to in a real deployment!)
4. **Priority: HIGH** — and also consider the deployment concern in point 3

> [!WARNING]
> **Potential methodological issue:** Your code computes drift direction using Batch 10 data (`b10 = df[df['Batch_ID']==10]['feat_0'].mean()`). In a real deployment, you wouldn't have future batch data at training time. Consider computing dir from Batch 1 → Batch 2 instead, or using domain knowledge about MOS sensor aging. A reviewer could flag this as data leakage if they look carefully.

---

### Comment 2.2.4 — GRL Reversal Coefficient

> *"The reversal coefficient (a key hyperparameter for GRL) is not specified."*

#### Your Reply

We thank the reviewer for noting this omission. The gradient reversal coefficient α is fixed at 1.0 throughout training. Unlike the original Domain-Adversarial Neural Network (DANN) framework [Ganin et al., 2016], which employs a progressive ramp-up schedule for α, we found that a constant coefficient provides stable convergence in our setting. This is attributed to the multi-loss regularization framework (7 loss terms in Phase 2) that already provides sufficient gradient stabilization, making the additional scheduling unnecessary. We have added this specification to the revised manuscript.

#### ✅ Recommended Action

1. **Add one sentence** to Section II.B: *"The gradient reversal coefficient is fixed at α = 1.0."*
2. Optionally cite [Ganin & Lempitsky, 2015] and briefly justify why scheduling was not needed
3. **Priority: LOW** — simple addition

---

## 2.3 Result Analysis and Discussion

---

### Comment 2.3.1 — Per-Sensor Performance Analysis

> *"Analyzing which sensors benefit most from the physics constraints would provide deeper insights..."*

#### Your Reply

We appreciate this thoughtful suggestion. The current PI-IDAN architecture processes the 128-dimensional feature vector holistically through the shared encoder, without explicit per-sensor decomposition. Consequently, per-sensor drift compensation analysis would require architectural modifications (e.g., sensor-wise attention mechanisms or separate encoding channels) that go beyond the scope of this letter.

However, we acknowledge this as a valuable research direction. In the revised manuscript, we have added a discussion note:

*"The current framework treats the sensor array response as a unified representation. Future work will explore sensor-wise attention mechanisms to identify which individual sensors contribute most to drift-invariant features and which benefit most from physics-informed constraints, enabling targeted sensor array optimization."*

#### ✅ Recommended Action

1. **Add 1–2 sentences** to the Conclusion/Future Work acknowledging this
2. **Do NOT attempt** a full per-sensor analysis for a 4-page letter — it would require significant new work and take too much space
3. If you want to go the extra mile: compute a simple feature importance analysis (e.g., gradient-based attribution for the 16 sensor groups of 8 features) and mention the result in one line
4. **Priority: LOW** — acknowledgment is sufficient

---

### Comment 2.3.2 — Ablation Study for Key Components

> *"Lacks a systematic ablation study for: physics head, memory replay, pseudo-labeling with Mean Teacher, supervised contrastive learning..."*

#### Your Reply

We agree with the reviewer that a systematic ablation study is essential for quantifying the contribution of each component. We have conducted a comprehensive ablation analysis with 5 independent seeds and present the results in Table X of the revised manuscript:

| Variant | Accuracy (%) | ΔAcc |
|---------|-------------|------|
| **PI-IDAN (Full)** | **XX.X ± X.X** | — |
| w/o Physics Head (Power Law + Monotonicity) | XX.X ± X.X | −X.X |
| w/o Monotonicity Loss only | XX.X ± X.X | −X.X |
| w/o Power Law Loss only | XX.X ± X.X | −X.X |
| w/o Mean Teacher (use direct pseudo-labels) | XX.X ± X.X | −X.X |
| w/o Supervised Contrastive Learning | XX.X ± X.X | −X.X |
| w/o Memory Replay | XX.X ± X.X | −X.X |

The ablation confirms that the physics head provides the largest individual contribution (−X.X% without it), followed by [component], validating the core thesis of this work.

#### ✅ Recommended Action

1. **This is the #1 most important addition to the revised paper**
2. You already have `run_ablation_study.py` — extend it to also ablate Mean Teacher and contrastive loss
3. Run with 5 seeds and report mean ± std
4. Add a compact table (will fit in ~6 lines of a column)
5. Add 2–3 sentences interpreting the results
6. **Priority: CRITICAL** — this single addition would have likely prevented the rejection

---

### Comment 2.3.3 — Late-Batch Performance Discussion (Batch 10: 57.6%)

> *"The manuscript should elaborate on the reasons for this drop and discuss potential improvements..."*

#### Your Reply

We thank the reviewer for raising this important practical concern. The performance degradation on Batch 10 (57.6% accuracy) is attributed to three compounding factors:

1. **Extreme drift magnitude:** Batch 10 was collected approximately 36 months after Batch 1. At this temporal distance, the cumulative drift exceeds the regime captured by the monotonic constraint, which was calibrated on early-batch baseline evolution. The linear monotonicity assumption (ReLU penalty on baseline change) may be insufficient for the non-linear drift acceleration observed at extreme timescales.

2. **Severe class imbalance:** Batch 10 exhibits significant class imbalance [Dennler et al., 2022], with certain gas classes dramatically underrepresented. This reduces the effectiveness of both contrastive learning (fewer positive pairs for minority classes) and pseudo-label quality (confidence-based filtering biases toward majority classes).

3. **Pseudo-label noise accumulation:** After 8 sequential adaptation steps (Batches 3–10), errors in pseudo-labels propagate and compound. Despite the Mean Teacher's temporal smoothing and ensemble agreement filtering, the accumulated noise degrades the classifier's decision boundaries for the final batch.

We have expanded the discussion in Section III.A and added potential improvements to the Conclusion, including adaptive non-linear drift models and explicit class-imbalance compensation during pseudo-label generation.

#### ✅ Recommended Action

1. **Add a dedicated paragraph** (4–5 sentences) in Section III.A discussing these three factors
2. **Add to Conclusion:** *"Future work will address extreme-drift scenarios through adaptive physics constraints with non-linear drift models, dynamic class-imbalance compensation, and confidence-calibrated pseudo-label refinement."*
3. Consider noting that 57.6% is still better than ALL baselines on Batch 10 (SVM: 38.9%, ISVM: 30.7%, 1DCNN: 31.5%, LSTM: 14.6%, IDAN: 54.3%) — this is a strong point you should emphasize!
4. **Priority: MEDIUM** — important but straightforward to write

---

### Comment 2.3.4 — Computational Overhead Analysis

> *"Does not report inference time or model size—critical for real-time sensor deployments..."*

#### Your Reply

We appreciate this practical suggestion. We have added a computational analysis to the revised manuscript:

| Component | Parameters |
|-----------|-----------|
| Siamese Encoder | ~XXX K |
| Task Classifier | ~X.X K |
| Physics Head | ~X.X K |
| Drift Discriminator | ~X.X K |
| **Total PI-IDAN** | **~XXX K** |

Inference requires only the encoder and classifier (~XXX K parameters total), as the physics head and discriminator are used only during training. The inference time is approximately X.X ms per sample on [hardware], making PI-IDAN suitable for real-time monitoring applications with typical sensor sampling rates of 0.1–10 Hz. Compared to LSTM-based approaches, the 1D-CNN encoder offers approximately X× faster inference due to its parallelizable convolutional operations.

#### ✅ Recommended Action

1. **Run the parameter counting script** (I can create this for you)
2. **Measure inference time** with a simple timing loop
3. Add a compact table or 2–3 sentences to Section III
4. Emphasize that at inference time, only encoder + classifier are needed (physics head and discriminator are training-only)
5. **Priority: MEDIUM** — easy to add, useful for practical impact

---

## 2.4 Presentation and Formatting

---

### Comment 2.4.1 — Figure Quality and Labeling

> *"Figure 2 has no axis labels, legend, or error bars; Figure 3 lacks legends; graphical abstract has typos..."*

#### Your Reply

We sincerely apologize for the substandard figure quality. All figures have been completely regenerated in the revised manuscript:

- **Figure 2 (Performance comparison):** Now includes clearly labeled x-axis (Batch ID), y-axis (Classification Accuracy %), legend differentiating PI-IDAN from baseline models, and error bars representing ± standard deviation across 5 seeds.
- **Figure 3 (t-SNE embedding):** Updated with explicit color legends for all 6 gas classes (Fig. 3a) and all 10 batch IDs (Fig. 3b), with clear marker differentiation.
- **Graphical abstract:** All typographical errors (e.g., "Yamazoe (S)AAa") have been corrected, and notation has been made consistent with the main text.

#### ✅ Recommended Action

1. **Completely redo Figure 2:**
   - X-axis: "Batch ID" (3–10)
   - Y-axis: "Classification Accuracy (%)"
   - Two lines: PI-IDAN and Baseline (IDAN) with different colors/markers
   - Error bars (±std from 5 seeds)
   - Legend with clear labels
   - Font size ≥ 10pt, 300 DPI minimum
2. **Redo Figure 3:**
   - Add color legends for gas classes and batch IDs
   - Use distinguishable markers/colors
3. **Fix graphical abstract:** Correct all typos, verify all equation notation
4. **Priority: HIGH** — bad figures are an instant credibility loss

---

### Comment 2.4.2 — Reference Formatting

> *"Inconsistent page number formatting (spaces in page numbers)..."*

#### Your Reply

We thank the reviewer for this observation. All references have been re-formatted to comply with IEEE Sensors Letters guidelines, with spaces in page numbers removed and journal/conference abbreviations standardized.

#### ✅ Recommended Action

1. Open `references.bib` and fix page number formatting
2. Specifically fix reference [8] (lee2024incremental): `pages={17286--17312}` — remove any spaces
3. Run BibTeX and verify output matches IEEE style
4. **Priority: LOW** — quick fix

---

### Comment 2.4.3 — Abstract and Conclusion Conciseness

> *"'Class-balanced memory replay' in abstract but not used in main text..."*

#### Your Reply

We have ensured terminology consistency throughout the revised manuscript. "Class-balanced memory replay" is now used consistently in both the abstract and the main text (Section II.B), where the class-balanced sampling mechanism is explicitly defined. The conclusion has been revised to focus on key contributions and practical implications rather than restating the architecture.

#### ✅ Recommended Action

1. Use "class-balanced memory replay" in both the abstract AND Section II.B
2. Trim the conclusion — remove architecture restatement, keep contributions + future work
3. **Priority: LOW** — simple consistency fix

---

## Master Priority Checklist

Here is your complete action list, ordered by priority:

### 🔴 CRITICAL (Must do before resubmission)

| # | Action | Effort |
|---|--------|--------|
| 1 | **Fix results accuracy** — re-run 5-seed experiments, report true mean±std | High |
| 2 | **Add ablation table** — extend and run `run_ablation_study.py` with 5 seeds for all components | High |
| 3 | **Fix ALL notation typos** — complete audit of every symbol in the paper | Medium |
| 4 | **Redo all figures** — proper axes, legends, error bars, 300 DPI | Medium |

### 🟡 HIGH (Strongly expected by reviewer)

| # | Action | Effort |
|---|--------|--------|
| 5 | **Add implementation details** — batch size, hardware, seeds, epochs | Low |
| 6 | **Describe memory replay buffer** — core + distilled, 50/class, balanced sampling | Low |
| 7 | **Clarify physics head** — S=\|\|z\|\|₂ vs b from MLP, fix diagram | Low |
| 8 | **Explain drift direction** — how dir is computed, global vs per-sensor | Low |
| 9 | **Add hyperparameter justification** — rationale for λ values + Phase 1 vs 2 | Low |
| 10 | **Fix graphical abstract** — correct all typos | Low |

### 🟢 MEDIUM (Will strengthen the paper)

| # | Action | Effort |
|---|--------|--------|
| 11 | **Elaborate Batch 10 discussion** — three factors, note still best among baselines | Low |
| 12 | **Add computational overhead** — parameter count + inference time | Low–Med |
| 13 | **Add dataset preprocessing details** — 128 features, Z-score, no outliers | Low |
| 14 | **Report GRL coefficient** — α=1.0, fixed | Low |

### 🔵 LOW (Nice to have)

| # | Action | Effort |
|---|--------|--------|
| 15 | **Fix reference formatting** — page number spaces | Trivial |
| 16 | **Consistent terminology** — "class-balanced memory replay" throughout | Trivial |
| 17 | **Per-sensor analysis** — acknowledge as future work only | Trivial |
| 18 | **Fix drift direction computation** — consider using Batch 1→2 instead of Batch 10 | Low |

---

> [!IMPORTANT]
> **Before starting revisions, I recommend re-running all experiments first (items 1 & 2).** The ablation results will influence what you write in several sections, and having accurate numbers is the foundation everything else builds on. Once you have clean 5-seed results and ablation data, the writing revisions (items 3–18) can be done in 1–2 days.

> [!TIP]
> **Regarding the drift direction issue (item 18):** Your current code uses Batch 10 to compute drift direction, which technically constitutes using future information. For the resubmission, I recommend changing this to compute direction from Batch 1 → Batch 2 (your source training data only). This makes the method fully causal and deployment-ready, and it's a one-line code change. This proactive fix shows the reviewers you thought carefully about real-world applicability.
