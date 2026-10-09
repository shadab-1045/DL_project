# Phase 6: Final Live Evidence Consolidation

This document strictly consolidates the read-only live findings for the adversarially robust face-identity verification pipeline. No Phase 5 artifacts, models, thresholds, or configurations were modified during this phase.

## 1. Experimental Scope
The Phase 6 experiments were designed to test the real-world inference behavior of the frozen Phase 5 models against live webcam inputs. The core focus was testing whether the offline adversarially trained C-Adv model could successfully identify a synthetic live Native InSwapper attack without requiring any re-calibration or retraining.

## 2. Excluded Trial
**Condition:** GENUINE
**Status:** **EXCLUDED — INVALID IDENTITY MATCH**
**Reason:** The individual who sat for the physical live genuine trial (a male operator) did not match the enrolled gallery identity (`sample_data/lfw/Angelina_Jolie/*.jpg`). Because the identities fundamentally mismatched in physical reality, the system correctly rejected the user. However, this constitutes a different-person rejection rather than a valid genuine-user acceptance test. Therefore, no genuine-user performance claims can be drawn from Phase 6.

## 3. Valid Live Conditions
Two valid live trials were completed:
1. **Different-Person Trial:** The non-enrolled physical operator presented to the webcam natively.
2. **Impersonation Trial:** The enrolled identity (Angelina Jolie) was used as the source, and the non-enrolled physical operator was used as the live target via the Native InSwapper engine.

---

## 4. Exact Measured Results

### A. DIFFERENT_PERSON Trial
- **N frames**: 10
- **Face detection rate**: 100% (10/10)
- **Model A similarity (min/max/mean)**: 0.127 / 0.154 / 0.137
- **Model A accept count**: 0
- **C-Adv invocation count**: 0
- **Mean C-Adv probability**: N/A
- **C-Adv predicted-class counts**: N/A
- **Final-state counts**: UNKNOWN (10)
- **Latency (min/max/mean)**: 0.452s / 2.507s / 1.669s
- **Approximate FPS**: 0.6 FPS

### B. IMPERSONATION Trial
- **N frames**: 10
- **Face detection rate**: 100% (10/10)
- **Model A similarity (min/max/mean)**: 0.779 / 0.796 / 0.788
- **Model A accept count**: 10
- **C-Adv invocation count**: 10
- **Mean C-Adv probability**:
  - `genuine`: 0.402
  - `different_person`: 0.566
  - `impersonation`: 0.032
- **C-Adv predicted-class counts**: Class 1 (`different_person`): 10
- **Final-state counts**: UNKNOWN (10)
- **Latency (min/max/mean)**: 3.695s / 4.909s / 4.346s
- **Approximate FPS**: 0.23 FPS

**Impersonation Defense Effectiveness:**
- **Percentage of frames Model A accepted synthetic identity**: 100%
- **Percentage of frames producing VERIFIED**: 0%
- **Percentage of frames producing SUSPECTED_IMPERSONATION**: 0%
- **Percentage of frames producing UNKNOWN**: 100%

---

## 5. Interpretation & Architectural Distinctions

**Identity-Gate Bypass:**
The Native InSwapper attack effectively bypassed the raw identity gate. Model A accepted 100% of the synthetic frames with extremely high confidence (mean 0.788), proving the danger of relying solely on identity recognition metrics.

**Safe Rejection / UNKNOWN:**
The system maintained operational security against the attack. By relying on C-Adv, it successfully prevented a false `VERIFIED` state on 100% of the impersonation frames. C-Adv recognized the input as non-genuine, safely downgrading the state to `UNKNOWN`.

**Explicit Impersonation Detection:**
The live system **did NOT achieve SUSPECTED_IMPERSONATION** on the tested Native InSwapper attack. The C-Adv model classified the generative artifacts as `different_person` rather than `impersonation`. It lacks the domain adaptation required to confidently assert the presence of synthetic face-swapping when confronted with this specific live generator.

## 6. Limitations
1. **Missing Genuine Evaluation:** The genuine condition was untestable without the physical presence of the enrolled subject. Genuine verification accuracy in live settings remains unknown.
2. **CPU Hardware Constraint:** Live latency was exceedingly high (up to 4.9s/frame) due to the CPU execution fallback in the Windows subprocess environment.
3. **Generator Gap:** The system was only tested live against `inswapper_128.onnx`. Generalization to other models is not represented here.

## 7. Artifact Integrity Verification
All live testing and diagnostic evaluations were strictly read-only. **No Phase 5 artifacts, models, thresholds, galleries, or configurations were modified at any point during Phase 6.**
