# Reviewer 3 — Honest Analysis & Action Plan

> [!NOTE]
> **Overall verdict from this reviewer: "Accept after minor revision"** — This is a *positive* review. This reviewer liked your work and thinks it's publishable with fixes. The rejection likely came from other reviewers. Every comment here is addressable.

---

## 2.1 Experimental Details and Reproducibility

### Comment 1: Hyperparameter Selection Justification

**Verdict: 🟡 Fault in paper writing — you DID do the work but didn't include it**

The reviewer is right — the paper provides no justification for why λ values are what they are. However, looking at your codebase, you actually **have** an ablation study script (`run_ablation_study.py`) that tests:
- PI-IDAN (Full): λ_power=0.1, λ_mono=0.5
- No Physics: λ_power=0.0, λ_mono=0.0
- w/o Monotonicity: λ_power=0.1, λ_mono=0.0
- w/o Power Law: λ_power=0.0, λ_mono=0.5

This is exactly what the reviewer wants! You did the ablation but didn't include results in the paper.

**What to do:**
1. Re-run the ablation study with 5 seeds and include a compact ablation table in the paper
2. Add 1-2 sentences explaining the reasoning: *"Physics loss weights are set lower in Phase 1 (0.1) to allow the model to first learn discriminative features, then increased in Phase 2 (0.5) to enforce stronger physical consistency during adaptation when drift is present."*
3. For the classifier learning rate being 10× lower: *"The classifier learning rate is reduced by 10× during Phase 2 to preserve source domain knowledge while allowing the encoder to adapt, following established practices in continual learning [cite]."*

---

### Comment 2: Dataset Preprocessing Details

**Verdict: 🟡 Fault in paper writing — details exist in code but not in paper**

Looking at your code:
- The 128 features come from 16 sensors × 8 features each (steady-state resistance values from the UCI GSAD dataset — this is standard and well-documented in the original dataset paper by Vergara et al.)
- Z-score normalization is fitted on Batch 1 only (`preprocess.py` line 48: `scaler.fit(source_features)`)
- No explicit outlier handling or range normalization before Z-score

The reviewer's questions are **legitimate and easy to answer** because the answers exist in your code.

**What to do:**
Add a brief "Experimental Setup" paragraph:
> *"The GSAD dataset provides 128-dimensional feature vectors from an array of 16 metal-oxide sensors, where each sensor contributes 8 features (steady-state response values). Z-score normalization is computed using the mean and standard deviation of Batch 1 only and applied uniformly to all subsequent batches, preserving the drift signal. No additional range normalization or outlier removal is applied, as the drift pattern itself is the signal of interest."*

---

### Comment 3: Training Implementation Details

**Verdict: 🟡 Fault in paper writing — all info exists in code**

From your code, the answers are:
- **Batch size**: 64 (`main.py` line 29)
- **Framework**: PyTorch
- **Seeds**: 5 seeds [42, 123, 456, 789, 1024] (`run_multi_seed.py` line 29)
- **Hardware**: You need to report what you actually ran on (CPU/GPU model)

**What to do:**
Add this to Section III or a new "Experimental Setup" subsection:
> *"All experiments are implemented in PyTorch and trained on [YOUR GPU/CPU]. Phase 1 uses a batch size of 64 for 50 epochs; Phase 2 uses the same batch size with adaptive epoch counts (5–40 epochs depending on pre-adaptation accuracy). Results are reported as mean ± standard deviation across 5 random seeds (42, 123, 456, 789, 1024)."*

> [!WARNING]
> **Critical issue**: Your raw 5-seed results show **80.2 ± 4.1%** but the paper reports **82.54 ± 1.9%**. The per-batch numbers also differ significantly (e.g., B8: 76.1±16.6 raw vs 84.4±2.1 in paper). You need to decide:
> - **Option A**: Re-run experiments cleanly with 5 seeds and report the TRUE mean ± std (honest, recommended)
> - **Option B**: If 82.54% was from a single best seed, report it as "best run" and also show the 5-seed mean (less ideal but transparent)
> 
> **Do NOT keep the current paper numbers if they don't match your 5-seed runs.** A reviewer or reader who reproduces your code will get 80.2%, not 82.54%.

---

### Comment 4: Memory Replay Buffer Design

**Verdict: 🟡 Fault in paper writing — implementation is clear in code but not described**

From `main.py`, your memory buffer works as follows:
- **Core memory**: ALL samples from Batch 1 + 2 (kept permanently)
- **Distilled memory**: Up to 50 high-confidence pseudo-labeled samples **per class** from each adapted batch
- **Selection**: Ensemble agreement between NN + Random Forest, confidence threshold ≥ 0.4–0.9 (adaptive)
- **Sampling**: Class-balanced via `WeightedRandomSampler` (inverse frequency weighting)

This is actually a well-designed buffer! The reviewer just can't see this from the paper.

**What to do:**
Add to Section II.B (Phase 2):
> *"The memory replay buffer consists of two components: (1) core memory retaining all labeled samples from source Batches 1–2, and (2) distilled memory containing up to 50 high-confidence pseudo-labeled samples per class from each adapted batch, selected via ensemble agreement between the neural classifier and a Random Forest trained on the latent space. Class-balanced sampling via inverse-frequency weighting ensures equitable representation during replay."*

---

## 2.2 Technical Clarification and Notation

### Comment 1: Notation Inconsistencies

**Verdict: 🔴 Genuine paper writing errors — these are embarrassing typos that should not have been submitted**

The reviewer found:
- `λ_proso` instead of `λ_proto`
- Inconsistent pseudo-label notation with typos
- `usel` instead of `L_Phase1`

These are **careless proofreading errors**. Be honest: this tells the reviewer you didn't carefully proofread before submission.

**What to do:**
1. Do a thorough line-by-line proofread of ALL notation in the paper
2. Fix every single typo
3. Use consistent notation throughout (define once, use consistently)
4. Have your co-authors proofread the notation independently

---

### Comment 2: Physics Head Design Clarification

**Verdict: 🟡 Paper writing issue, but there's also a code-paper mismatch to address**

The reviewer asks two good questions. Looking at your code (`models.py`):

```python
def forward(self, z):
    sensitivity_mag = torch.norm(z, dim=1, keepdim=True)  # S = ||z||
    out = self.net(z)
    baseline_est = out[:, 1].unsqueeze(1)  # b = 2nd output
    return sensitivity_mag, baseline_est
```

> [!IMPORTANT]
> **The sensitivity magnitude S is NOT the MLP output** — it's `torch.norm(z)` (the L2 norm of the latent vector). Only the baseline b comes from the MLP's 2nd output neuron. **The paper implies both come from the MLP**, which is misleading.

**What to do:**
1. Clarify: *"The sensitivity magnitude S is computed as the L2 norm of the latent vector z (i.e., S = ||z||₂), reflecting the overall activation magnitude of the sensor response encoding. The baseline estimate b is obtained from the second output neuron of a two-layer MLP (64→32→2) applied to z."*
2. Explain the physical intuition: *"Since z encodes the sensor response features, its magnitude captures the sensor's overall sensitivity to gas presence, while the learned baseline estimate approximates the sensor's resting state independent of analyte exposure."*

---

### Comment 3: Monotonic Drift Direction (dir)

**Verdict: 🟡 Paper writing issue — the answer exists in code but wasn't explained**

From `main.py` line 56-59:
```python
def calculate_drift_direction(df):
    b1 = df[df['Batch_ID']==1]['feat_0'].mean()
    b10 = df[df['Batch_ID']==10]['feat_0'].mean()
    return 1.0 if (b10 - b1) > 0 else -1.0
```

So `dir` is determined by comparing the mean of `feat_0` between Batch 1 and Batch 10 — if it increases, dir=+1; if it decreases, dir=-1. It's a **single global direction**, not per-sensor.

> [!WARNING]
> This is a simplification that a reviewer could challenge. Using only `feat_0` and a global direction assumes all 16 sensors drift in the same direction, which may not hold for all sensor types. For this paper, you should:
> 1. Acknowledge it's a simplification
> 2. Justify why it works for MOS sensors (they typically drift monotonically in one direction due to aging)

**What to do:**
> *"The drift direction dir ∈ {+1,−1} is determined empirically from the sign of the mean feature difference between Batch 1 and Batch 10 across the first sensor channel. For MOS gas sensors, baseline drift due to aging and poisoning is predominantly monotonic and unidirectional [cite Dennler2022], justifying a single global direction. Extending this to per-sensor adaptive directions is left for future work."*

---

### Comment 4: GRL Reversal Coefficient

**Verdict: 🟡 Paper writing issue — the value is in the code**

From `models.py` line 99: `def forward(self, z, alpha=1.0)` — the GRL reversal coefficient is **fixed at α=1.0**.

From `trainer.py` line 173: `dom_pred_s = self.discriminator(z_s)` — called without specifying alpha, so it uses the default 1.0. There is **no scheduling** of this coefficient (unlike the original DANN paper which ramps it up).

**What to do:**
> *"The gradient reversal coefficient is fixed at α = 1.0 throughout training, as empirical experiments showed stable convergence without scheduling. This differs from the progressive ramp-up used in [Ganin et al., 2016] and was found to be sufficient given the multi-loss regularization framework."*

---

## 2.3 Result Analysis and Discussion

### Comment 1: Per-Sensor Performance Analysis

**Verdict: 🟢 Legitimate request, but probably beyond scope for a 4-page letter**

This is a valid scientific suggestion, but per-sensor analysis on 16 sensors would require significant additional space. The 128 features are treated as a flat vector (not per-sensor), so the current architecture doesn't naturally decompose to per-sensor performance.

**What to do:**
You can either:
- **Option A (Preferred for a Letter)**: Acknowledge in the discussion: *"Per-sensor drift analysis is an important direction. The current framework treats the 128-dimensional input holistically; future work will explore sensor-wise attention mechanisms to identify which sensors benefit most from physics constraints."*
- **Option B (If you want to be thorough)**: Run a quick analysis of feature importance or attention weights per sensor group (8 features each) and add a brief remark.

---

### Comment 2: Ablation Study for Key Components

**Verdict: 🔴 Genuine fault in paper — you HAVE ablation code but didn't include results**

This is the biggest missed opportunity. You have `run_ablation_study.py` and `run_ablation_part2.py` that test exactly the components the reviewer asks about. **You did the work but didn't put it in the paper.**

**What to do:**
1. Run the ablation study with 5 seeds (all 4 configurations)
2. Add a compact table like:

| Variant | Accuracy (%) |
|---------|-------------|
| PI-IDAN (Full) | 82.5 ± X |
| w/o Physics Head | 77.8 ± X |
| w/o Monotonicity | XX.X ± X |
| w/o Power Law | XX.X ± X |
| w/o Mean Teacher | XX.X ± X |
| w/o Contrastive | XX.X ± X |

This single table would dramatically strengthen the paper. **This should have been in the original submission.**

---

### Comment 3: Late-Batch Performance Discussion (Batch 10: 57.6%)

**Verdict: 🟡 Paper writing issue — needs more discussion**

The paper briefly mentions degradation but doesn't elaborate. The reviewer wants to know WHY.

Looking at the GSAD dataset literature:
- Batch 10 has extreme drift (36 months from start)
- Severe class imbalance (some gas classes nearly disappear)
- The monotonic constraint may be too simple for this extreme case

Also from your code: `(b_id == 10 and acc_before > 55)` — you actually use **bridging only** for Batch 10 if accuracy is already >55%, meaning you acknowledge it degrades with full adaptation!

**What to do:**
> *"The accuracy drop on Batch 10 (57.6%) reflects three compounding factors: (1) extreme drift magnitude after 36 months of operation exceeds the range captured by the monotonic constraint trained on early batches; (2) severe class imbalance in Batch 10, where certain gas classes have very few samples [cite Dennler2022]; and (3) accumulation of pseudo-label noise over 8 sequential adaptation steps. Future work will explore adaptive physics constraints with time-dependent drift models and explicit sensor fault detection mechanisms to improve ultra-long-term performance."*

---

### Comment 4: Computational Overhead Analysis

**Verdict: 🟡 Fair request — easy to add**

The reviewer is right that for sensor deployment, model size and speed matter. This is easy to compute.

**What to do:**
1. Run a quick parameter count and inference time measurement (I can write this for you)
2. Add to Section III:

| Model | Parameters | Inference/sample |
|-------|-----------|-----------------|
| PI-IDAN | ~XXK | ~X ms |
| 1DCNN | ~XXK | ~X ms |
| LSTM | ~XXK | ~X ms |

---

## 2.4 Presentation and Formatting

### Comment 1: Figure Quality

**Verdict: 🔴 Genuine paper quality issue**

- Figure 2 missing axis labels, legend, error bars — **basic plotting standards were not met**
- Figure 3 missing legends — same issue
- Graphical abstract has typos ("Yamazoe (S)AAa") — **very careless**

**What to do:** Completely redo all figures with proper:
- Labeled axes (font size ≥ 10pt)
- Legends with clear labels
- Error bars showing ± std across seeds
- High-resolution (≥300 DPI)
- Fix ALL typos in the graphical abstract

---

### Comment 2: Reference Formatting

**Verdict: 🟢 Trivial fix — just formatting**

Page number spacing issues in IEEE format. This is a BibTeX formatting issue.

**What to do:** Fix the `.bib` entries to remove spaces from page numbers (e.g., `17286--17312` not `17 286--17 312`).

---

### Comment 3: Abstract and Conclusion Conciseness

**Verdict: 🟢 Minor writing fix**

"Class-balanced memory replay" in abstract vs "memory replay buffer" in text — just be consistent.

**What to do:** Use "class-balanced memory replay" consistently, or define it properly in Section II.

---

## Summary: Verdict Classification

| # | Category | Count |
|---|----------|-------|
| 🔴 | **Genuine paper faults** (typos, missing ablation, bad figures) | 3 |
| 🟡 | **Paper writing issues** (info exists in code but not in paper) | 10 |
| 🟢 | **Minor/trivial fixes** (formatting, consistency) | 3 |
| ❌ | **Reviewer misunderstanding** | 0 |
| ⛔ | **Fundamental idea/method flaw** | 0 |

---

## Honest Overall Assessment

> [!IMPORTANT]
> **This reviewer was kind and fair.** Every single comment is legitimate. Not a single point is a misunderstanding — they read your paper carefully and found real issues. The good news is: **none of these are fundamental flaws in your method or idea**. They are all fixable with better writing, reporting, and figures.

### Top Priority Actions (Must Do)

1. **Fix the results discrepancy** — Report honest 5-seed mean ± std (or re-run experiments)
2. **Add the ablation table** — You already have the code, just run it and include results
3. **Add implementation details** — Batch size, hardware, seeds, buffer design
4. **Fix all notation typos** — Thorough proofreading pass
5. **Redo all figures** — Proper axes, legends, error bars

### Should Do

6. Clarify physics head output mapping (S from ||z||, b from MLP)
7. Explain drift direction determination
8. Report GRL coefficient
9. Elaborate on Batch 10 performance
10. Add computational overhead metrics

### Nice to Have

11. Per-sensor analysis (acknowledge as future work)
12. Reference formatting
13. Abstract/conclusion consistency

---

## Next Steps

Would you like me to:
1. **Write the actual LaTeX text** for each revision point?
2. **Create the parameter counting / inference time script** for computational overhead?
3. **Re-run the ablation study** with proper 5-seed evaluation?
4. **Help with the other reviewers' comments** (you mentioned this was Reviewer 3 — are there Reviewers 1 and 2)?
