# Designing Ambiguity Aware ClericalReview

Code to accompany article submission.

# Record-level conditional Perplexity & Matchability

## Overview
- **Goal:** quantify *how certain* a linkage decision is **per record**, and summarise uncertainty for **sub-populations** and the **whole dataset**.  

- **Outputs per record:**  
  - `p_null` = probability of **no match**; `matchability = 1 − p_null`  
  - `perp_cond` = **effective number of plausible candidates** *given it links*  
  - optional: `H_cond` (bits), `top1`, `margin`, candidate-level `p_i`, `p_cond`
- **Pre-linkage ambiguity:** count how many records fall into the **same evidence neighbourhood** (built from your comparison levels); report `Perp_pre = n_pre` and `H_pre = log2(n_pre)`.  
- **Nice combo:** **Resolution Gain** = `H_pre − H_cond` shows how much our linker resolved the crowding.

---

## 1) Why this is needed (pairwise ≠ record-level)

Splink’s `predict()` returns **pairwise** scores (`match_weight`, `match_probability`) for each **left record × candidate** edge. For record-level decisions you need a **single distribution** over mutually exclusive outcomes for that record:

> {match to candidate 1, 2, …, **no match**}.

From this we derive:
- **`p_null`** (unlinkable probability)  
- **`matchability = 1 − p_null`**  
- **`perp_cond`** (ambiguity among the candidates *if* it does link)

These drive **clerical review** and are easy to aggregate by **sub-group** and **population**.

---

## 2) The math (weights *or* probabilities)

For one left record with candidates \(i=1..K\):

### 2.1 Convert to odds vs no-match
- From Splink **weights** \(Mi\) (base-2 log-odds):  
  <img width="227" height="72" alt="image" src="https://github.com/user-attachments/assets/147af06d-294c-4621-84a4-6c443348a66b" />

- From pairwise **probabilities** \(qi\):  
  <img width="230" height="82" alt="image" src="https://github.com/user-attachments/assets/8d2c56f8-061c-4311-a0ef-e2fef3545e69" />

> **Note on prior:** Splink’s *linkable prior* is already inside \(Mi\) / \(qi\).  
> When forming the record-level distribution, set the no-match odds to **1** (neutral), unless you intentionally re-prior.

### 2.2 Normalise over {candidates + no-match}
<img width="609" height="119" alt="image" src="https://github.com/user-attachments/assets/20d8cd23-affc-4a55-b011-e6b9581469bc" />

### 2.3 Record-level metrics
- **Matchability: <img width="190" height="51" alt="image" src="https://github.com/user-attachments/assets/99e79966-7380-4b20-9676-7cbaca6bfdaf" />
- **Conditional candidate probabilities (given it links):**  
  <img width="274" height="64" alt="image" src="https://github.com/user-attachments/assets/e7049868-672e-4db1-9bf1-5f9b51ce6927" />

- **Conditional entropy (bits) & perplexity:**  
  <img width="274" height="73" alt="image" src="https://github.com/user-attachments/assets/26170d43-8afd-42f5-8fef-56f0f2eb98ff" />
  (≈ **effective number of plausible candidates**; continuous, so 1.03 means “almost unique”.)
- (Optional) **Unconditional** entropy/perplexity (include `p_null`) for dataset QA.



## 3) Calibration & priors (short)

- **Calibration matters:** if pairwise scores are over/under-confident, entropy/perplexity will be biased. 
- **Priors:** keep the default \(O_0=1\) (Splink’s prior already applied). Override only if you intentionally re-prior for a specific dataset/source.


