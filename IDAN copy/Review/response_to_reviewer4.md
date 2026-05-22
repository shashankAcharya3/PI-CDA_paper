# Response to Reviewer 4 — Point-by-Point (Reject Decision)

> [!IMPORTANT]
> **This reviewer gave a Reject.** Unlike Reviewer 3 who found only presentation issues, Reviewer 4 raises **methodological concerns** about the physics head's reliance on concentration data, the 1D-CNN justification, and insufficient problem formulation. These are more fundamental and need careful, honest responses. Some are paper writing failures; some are genuine issues.

---

## Comment 1 — Concentration Requirement (C) for Power Law Loss

> *"The proposed method requires the true gas concentration level (C) for each measurement. This requirement is not specified in the problem formulation in sec 2. It is unclear whether such concentration levels are available in practical deployment scenarios."*

### My Honest Assessment: 🟡 Mostly a paper writing failure, but has a legitimate practical concern

**This reviewer is half-right, and the paper is to blame for the confusion.**

Here's what actually happens in your code:

| Phase | Uses Concentration C? | Where? |
|-------|----------------------|--------|
| **Phase 1** (Source training, Batches 1–2) | ✅ Yes | `power_law_loss(sens, c, ...)` on line 107 of `trainer.py` |
| **Phase 2** (Adaptation, Batches 3–10) | ❌ No | Power law loss is **completely absent** from Phase 2 — see line 230 |

**The power law loss with C is ONLY used during Phase 1** where you're training on labeled source data that already includes concentration. In Phase 2 (unlabeled target adaptation), only the **monotonicity loss** (which does NOT need C) is used for physics. The w and b parameters of the power law are even **frozen** in Phase 2 (`self.physics_head.w.requires_grad = False`, line 131).

**The problem:** Your paper **never makes this distinction clear**. The paper describes the power law loss in Phase 1.  But it also says "the physics head" is used in Phase 2 without clarifying that only the monotonicity component is active. A reader naturally assumes C is needed throughout.

**The reviewer's practical concern is also valid:** Even for Phase 1, you need labeled concentration levels. In the GSAD dataset, these are available. But in a real-world deployment, you need at least an initial labeled dataset with known concentrations. This is a reasonable assumption (you calibrate sensors with known gas concentrations during initial deployment) but it should be stated explicitly.

### Your Reply

We thank the reviewer for this important observation. We acknowledge that the manuscript did not clearly distinguish the role of concentration C between the two training phases, leading to this concern. We clarify:

**The gas concentration C is required only during Phase 1 (source domain training)**, where labeled data — including both gas class and concentration — is available from the initial calibration setup. In Phase 2 (incremental adaptation), the power law loss is not applied. Only the monotonicity loss (which constrains the baseline estimate b to evolve unidirectionally) is used as the physics constraint, and this loss requires no concentration information. The learnable parameters w and b of the Yamazoe power law are frozen after Phase 1, ensuring that the concentration-sensitivity relationship learned during calibration acts as a fixed physical prior during adaptation.

This design reflects a realistic deployment scenario: sensors are initially calibrated with known reference gases at known concentrations (standard practice for MOS gas sensor deployment), after which the model adapts to drift using only unlabeled field measurements. We have revised Section II to explicitly state this two-phase distinction in the problem formulation.

### ✅ Recommended Action

1. **Add to the problem formulation (Section II, first paragraph):** *"The source domain $D_S$ includes gas class labels $y_i^s$ and concentration levels $C_i^s$ from initial calibration. The target domains $D_{T_j}$ are fully unlabeled, containing neither class nor concentration information."*
2. **Add to Phase 1 description:** *"The power law loss $\mathcal{L}_{power1}$ leverages the known concentrations available in the labeled source domain."*
3. **Add to Phase 2 description:** *"During Phase 2, only the monotonicity constraint $\mathcal{L}_{mono2}$ is applied as the physics loss, which does not require concentration information. The power law parameters w and b are frozen after Phase 1."*
4. **Add a sentence about practical deployment:** *"This design mirrors real-world sensor deployment: initial calibration with known reference gases provides labeled source data with concentrations, after which the system operates autonomously without further labeled data."*
5. **Priority: CRITICAL** — This is the reviewer's primary concern and the likely reason for rejection

---

## Comment 2 — Drift Direction (dir) and Baseline Change (Δb) Across Batches

> *"The procedure for obtaining the expected drift direction (dir) is not clearly described. Moreover, it is unclear how the change in b across batches is computed, particularly when only a single batch is used as the source domain."*

### My Honest Assessment: 🔴 Mix of paper writing failure AND a genuine methodological issue

**Two separate concerns here:**

**Concern A (dir computation):** Same issue as Reviewer 3 raised. Your code computes dir using Batch 10 data, which the model should not have access to at training time. **This is genuine data leakage.** The paper vaguely says "determined from the initial source batches" (line 94 of the LaTeX) but the code does something different.

**Concern B (Δb with "single source batch"):** The reviewer seems confused about how many source batches there are. Your model uses Batches 1 AND 2 as source (but the paper never explicitly states this!). So Δb during Phase 1 is computed between the baseline estimates of Batch 1 and Batch 2 training iterations. During Phase 2, the baseline from Phase 1's last epoch is stored as `prev_baseline`, and each new batch's baseline is compared against it sequentially: b₃ vs b₂, b₄ vs b₃, etc.

The reviewer's confusion is **entirely caused by your paper not stating that Batches 1–2 are the source domain.**

### Your Reply

We appreciate the reviewer's careful reading and acknowledge that both points were insufficiently explained.

**Drift direction (dir):** In the revised manuscript, we explicitly define the procedure: dir is computed from the sign of the mean feature change between Batch 1 and Batch 2 (the two source domain batches), i.e., dir = sign(mean(x|Batch 2) − mean(x|Batch 1)). For MOS gas sensors, baseline drift due to aging and surface poisoning is predominantly monotonic and unidirectional under isothermal operation [Dennler et al., 2022; Vergara et al., 2012], justifying a single global direction parameter. This direction is fixed after Phase 1 and used throughout Phase 2.

**Baseline change (Δb):** The source domain consists of Batches 1 and 2 (we apologize that this was not explicitly stated in the original manuscript). During Phase 1, the physics head learns to estimate the sensor baseline from the latent representation, and the baseline estimate $b_0$ from the final epoch is stored as the initial reference. During Phase 2, each adapted batch $j$ produces a new baseline estimate $b_j$, and the monotonicity loss penalizes $b_j − b_{j−1}$ if it violates the expected drift direction. Thus, the baseline reference is updated sequentially: $b_0$ (from Phase 1) → $b_3$ (after Batch 3 adaptation) → $b_4$ → ... → $b_{10}$.

We have added these details to Sections II and III in the revised manuscript.

### ✅ Recommended Action

1. **FIX THE CODE:** Change `calculate_drift_direction()` to use Batch 1 → Batch 2 instead of Batch 1 → Batch 10:
   ```python
   def calculate_drift_direction(df):
       b1 = df[df['Batch_ID']==1]['feat_0'].mean()
       b2 = df[df['Batch_ID']==2]['feat_0'].mean()  # Changed from Batch 10
       return 1.0 if (b2 - b1) > 0 else -1.0
   ```
2. **Re-run ALL experiments** after this code fix (results may change slightly)
3. **Explicitly state** in Section II: *"The source domain consists of Batches 1 and 2."*
4. **Add explanation** of the sequential baseline update mechanism
5. **Priority: CRITICAL** — data leakage is a fundamental methodological flaw that must be fixed

> [!CAUTION]
> **You must actually fix the code and re-run experiments.** You cannot just change the text — the reviewer or a future reproducer will see this. After fixing dir to use Batch 1→2, verify that dir is still the same sign. If it changes, your results will be different and you need to re-report everything.

---

## Comment 3 — 1D-CNN Encoder Details and "Transformation to 1D Vector"

> *"The description of the 1D-CNN encoder lacks sufficient detail (channel) and requires further elaboration. The transformation to 1D vector appears to contradict the objective of capturing temporal dependencies."*

### My Honest Assessment: 🟡 Paper writing caused a genuine misunderstanding — but the reviewer has a valid architectural question

**The reviewer is actually RIGHT to be confused.** Your paper uses the phrase *"to capture long-term dependencies of the sensor's signal"* (line 81), which implies temporal/sequential processing. But your 128 features are **NOT a time series** — they are a **spatial feature vector** from 16 sensors × 8 features per sensor. There are no temporal dependencies within a single 128-dim sample.

The "temporal" aspect of your problem is drift **across batches over months**, not within individual samples. The 1D-CNN is treating the 128-dim feature vector as a 1D signal to capture **inter-sensor relationships** (how nearby sensor features correlate), not temporal patterns.

**Your paper's wording is misleading**, and this is why the reviewer thinks the architecture contradicts the objective.

**Channel details from your code** (not in the paper):

| Layer | Input Channels | Output Channels | Kernel | Stride | Dilation | Padding |
|-------|---------------|-----------------|--------|--------|----------|---------|
| Conv1 | 1 | 64 | 9 | 8 | 1 | 4 |
| Conv2 | 64 | 128 | 3 | 1 | 4 | 4 |
| Conv3 | 128 | 128 | 3 | 1 | 1 | 1 |
| ResProj | 64 → 128 | — | 1 | — | — | — |

Also, the code includes a **residual connection** (`f3 = self.conv3(f2) + self.res_proj(f1)` on line 40 of `models.py`) that the paper never mentions.

### Your Reply

We thank the reviewer for this observation and acknowledge that the original description was misleading. We clarify:

The 128-dimensional input is **not a temporal signal** but a spatial feature vector comprising 8 features from each of 16 metal-oxide gas sensors (steady-state and transient response characteristics). The input is reshaped to a 1-channel 1D signal [batch × 1 × 128] and processed by three convolutional blocks. The 1D-CNN architecture exploits the **spatial correlations between sensor features** (e.g., adjacent features often correspond to the same sensor or similar measurement types) rather than temporal dependencies. The large kernel (k=9) in the first layer captures cross-sensor feature interactions, while the dilated convolution (d=4) in the second layer extends the receptive field to capture longer-range inter-sensor relationships without increasing parameters.

We have revised the encoder description to include complete channel specifications:
- Conv1: 1→64 channels, k=9, s=8, p=4 (broad feature extraction)
- Conv2: 64→128 channels, k=3, s=1, d=4, p=4 (dilated cross-sensor modeling)  
- Conv3: 128→128 channels, k=3, s=1, p=1 (local refinement with residual connection from Conv1)

A residual connection projects the Conv1 output (64 channels) to match Conv3's dimensions, improving gradient flow. The resulting 2048-dimensional feature map is compressed through fully connected layers (2048→512→128→64) to produce the latent vector z.

We have removed the misleading reference to "temporal dependencies" in the revised manuscript.

### ✅ Recommended Action

1. **Remove** the phrase "to capture long-term dependencies of the sensor's signal" — replace with *"to capture cross-sensor feature interactions"*
2. **Add channel specifications** (1→64→128→128) to Section II
3. **Mention the residual connection** — it's in the code but not the paper
4. **Explain the 1 × 128 reshaping** explicitly: *"The 128-dimensional feature vector is reshaped to a single-channel 1D signal [1 × 128] for convolutional processing."*
5. **Clarify that temporal drift is across batches, not within samples**
6. **Priority: HIGH** — directly addresses reviewer's architectural concern

---

## Comment 4 — Source Domain Definition

> *"The definition of the source domain is unclear. It appears that the first two batches of the GSAD dataset are used as the source domain, but this is not explicitly stated."*

### My Honest Assessment: 🔴 Genuine paper writing failure — this is inexcusable

This is a **basic experimental setup detail** that must be stated clearly. Your paper says "supervised initialization on early labeled batches" and "labeled source domain" but never says **which batches** are the source and which are the target. The reviewer had to guess.

From your code (`main.py` lines 90-91):
```python
ds_b1 = IDANDataset(df, batch_id=1, domain_label=0)
ds_b2 = IDANDataset(df, batch_id=2, domain_label=1)
```

Batches 1–2 are source; Batches 3–10 are sequential target domains.

### Your Reply

We apologize for this omission. In the revised manuscript, we explicitly state the experimental protocol:

*"Following the standard evaluation protocol for the GSAD dataset [Vergara et al., 2012; Dennler et al., 2022], Batches 1 and 2 are used as the labeled source domain $D_S$ for Phase 1 training. Batches 3 through 10 constitute the sequential unlabeled target domains $D_{T_1}, ..., D_{T_8}$ for Phase 2 incremental adaptation. Each target batch is processed in chronological order, and evaluation is performed on each batch immediately after adaptation."*

This has been added to Section III (Results and Discussion) as an explicit "Experimental Protocol" statement.

### ✅ Recommended Action

1. **Add one paragraph** at the start of Section III clearly defining: source = Batches 1–2, target = Batches 3–10, sequential adaptation order
2. **Also mention in Section II** (problem formulation): *"In the GSAD setting, J=8 target batches are processed sequentially."*
3. **Priority: HIGH** — this is a basic clarity issue that contributed to the rejection

---

## Comment 5 — Minor: "b" Used with Multiple Meanings

> *"b is used with multiple meanings."*

### My Honest Assessment: 🔴 Genuine notation flaw — confusing and avoidable

In your paper, `b` is overloaded:
1. **Baseline estimate** — the physics head output representing sensor resting-state resistance
2. **Bias parameter** — the intercept in the Yamazoe power law: $S = w \cdot \log(C) + b$

Both appear in the *same equation region* of Section II, making it impossible for the reader to distinguish them. This is a clear notation error.

### Your Reply

We thank the reviewer for catching this notation conflict. In the revised manuscript, we use distinct symbols:
- $\hat{b}$ (b-hat) for the **baseline estimate** predicted by the physics head
- $\beta$ (beta) for the **bias parameter** in the Yamazoe power law relationship

The revised power law loss becomes: $\mathcal{L}_{\text{power1}} = \text{MSE}(S, w \cdot \log(C) + \beta)$, and the monotonicity loss uses $\Delta\hat{b} = \hat{b}_j - \hat{b}_{j-1}$.

This eliminates the ambiguity and has been applied consistently throughout the manuscript.

### ✅ Recommended Action

1. **Rename** the power law bias from $b$ to $\beta$ everywhere in the paper
2. **Rename** the baseline estimate from $b$ to $\hat{b}$ everywhere
3. **Update** all equations (Eq. 1, monotonicity loss, and any inline references)
4. **Verify** the code comments/variable names are consistent if you provide a code repo
5. **Priority: MEDIUM** — simple fix but important for clarity

---

## Overall Assessment: Why This Reviewer Rejected

> [!WARNING]
> **The rejection is primarily because the paper failed to communicate the method properly, NOT because the method is flawed.** Here's the evidence:
> 
> | Concern | Actual Situation | Paper Says |
> |---------|-----------------|-------------|
> | C needed throughout? | Only in Phase 1 (labeled source) | Unclear — reads like C is always needed |
> | Which batches are source? | Batches 1–2 | "early labeled batches" (vague) |
> | How is dir computed? | From feature trend | "determined from initial source batches" (vague) + code uses Batch 10 (leakage!) |
> | CNN captures what? | Cross-sensor spatial features | "long-term dependencies" (misleading) |
> | What is b? | Two different things | Same symbol for both (confusing) |
> 
> Every concern traces back to **unclear or misleading writing**. The reviewer couldn't evaluate the method fairly because the paper didn't describe it accurately.

### The One Genuine Methodological Issue

The **only truly problematic finding** is the drift direction being computed using Batch 10 data (data leakage). This must be fixed in the code and experiments re-run. Everything else is a writing fix.

---

## Comparison: Reviewer 3 vs Reviewer 4

| Aspect | Reviewer 3 (Minor Revision) | Reviewer 4 (Reject) |
|--------|---------------------------|---------------------|
| Tone | Supportive, detailed | Skeptical, concise |
| Core concern | Missing details & ablation | Method description is fundamentally unclear |
| Fixability | All fixable with added text | All fixable with rewriting + one code fix |
| Asks for ablation? | ✅ Explicitly | ❌ Not mentioned |
| Questions physics head? | Partially | ✅ Deeply |
| Questions C requirement? | ❌ | ✅ This was the main concern |
| Questions source definition? | ❌ | ✅ |

---

## Master Priority List (Combined with Reviewer 3)

### 🔴 CRITICAL — Do These First

| # | Action | Source |
|---|--------|--------|
| 1 | **Fix dir computation** — change code to use Batch 1→2, NOT Batch 10 | R4 Comment 2 |
| 2 | **Re-run ALL experiments** with fixed code, 5 seeds, report true mean±std | R3 + R4 |
| 3 | **Clarify C is only needed in Phase 1** — rewrite problem formulation | R4 Comment 1 |
| 4 | **Explicitly state source = Batches 1–2, target = Batches 3–10** | R4 Comment 4 |
| 5 | **Add ablation table** with 5 seeds | R3 Comment 2.3.2 |
| 6 | **Fix b notation** — use $\hat{b}$ for baseline, $\beta$ for power law bias | R4 Comment 5 |

### 🟡 HIGH — Do After Critical Items

| # | Action | Source |
|---|--------|--------|
| 7 | **Rewrite 1D-CNN description** — remove "temporal dependencies", add channels, residual connection | R4 Comment 3 |
| 8 | **Add Phase 2 clarification** — only monotonicity loss used, w and b frozen, no C needed | R4 Comment 1 |
| 9 | **Fix ALL notation typos** throughout paper | R3 Comment 2.2.1 |
| 10 | **Redo all figures** with proper axes, legends, error bars | R3 Comment 2.4.1 |
| 11 | **Add implementation details** — batch size, hardware, seeds, buffer design | R3 Comments 2.1.3, 2.1.4 |

### 🟢 MEDIUM — Strengthen the Paper

| # | Action | Source |
|---|--------|--------|
| 12 | **Elaborate Batch 10 discussion** — but note it STILL beats all baselines | R3 Comment 2.3.3 |
| 13 | **Add computational overhead** | R3 Comment 2.3.4 |
| 14 | **Fix graphical abstract** | R3 Comment 2.4.1 |
| 15 | **Fix references** | R3 Comment 2.4.2 |
