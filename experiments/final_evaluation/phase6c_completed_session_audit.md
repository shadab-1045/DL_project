# Phase 6C: Completed Live Session Audit

This document audits the completed Phase 6C manual live validation session logs found in `experiments/live_demo/manual/`.

## 1. Scenario: GENUINE Condition

**Observed Measurements:**
- **Log File**: `manual_genuine_1791043256.jsonl`
- **Frame count**: 10
- **Face-detection count**: 10
- **Model A similarity (min/max/mean)**: 0.117 / 0.164 / 0.139
- **Model A acceptance count**: 0 (Threshold: 0.244529)
- **C-Adv invocation count**: 0 (Bypassed due to Model A rejection)
- **Final-state counts**: UNKNOWN (10)
- **Latency (min/max/mean)**: 0.276s / 2.521s / 1.802s

**Visual Inspection & Validity:**
- **Diagnostic Image**: `diagnostic_genuine.jpg`
- **Validity**: **INVALID TRIAL**. The physical participant pictured in the diagnostic image is clearly NOT the enrolled identity (Alice/Angelina Jolie). Because the person presenting themselves to the camera was a different person, this trial fundamentally acted as a second "Different Person" evaluation. 
- **Interpretation**: Model A successfully rejected the non-enrolled user with low similarity. The pipeline correctly bypassed C-Adv and mapped the final state to `UNKNOWN`.

---

## 2. Scenario: DIFFERENT_PERSON Condition

**Observed Measurements:**
- **Log File**: `manual_different_person_1791043290.jsonl`
- **Frame count**: 10
- **Face-detection count**: 10
- **Model A similarity (min/max/mean)**: 0.127 / 0.154 / 0.137
- **Model A acceptance count**: 0
- **C-Adv invocation count**: 0
- **Final-state counts**: UNKNOWN (10)
- **Latency (min/max/mean)**: 0.452s / 2.507s / 1.669s

**Visual Inspection & Validity:**
- **Diagnostic Image**: `diagnostic_different_person.jpg`
- **Validity**: **VALID TRIAL**. The physical participant is not the enrolled identity, perfectly matching the required condition.
- **Interpretation**: Model A successfully rejected the non-enrolled user, keeping the system secure and yielding `UNKNOWN`.

---

## 3. Scenario: IMPERSONATION Condition

**Observed Measurements:**
- **Log File**: `manual_impersonation_1791043312.jsonl`
- **Frame count**: 10
- **Face-detection count**: 10
- **Model A similarity (min/max/mean)**: 0.779 / 0.796 / 0.788
- **Model A acceptance count**: 10
- **C-Adv invocation count**: 10
- **C-Adv Mean Probabilities**:
  - `genuine`: 0.402
  - `different_person`: 0.566
  - `impersonation`: 0.032
- **C-Adv Predicted Classes**: Class 1 (`different_person`): 10
- **Final-state counts**: UNKNOWN (10)
- **Latency (min/max/mean)**: 3.695s / 4.909s / 4.346s

**Comparison with Controlled Diagnostic:**
- **Diagnostic**: `[genuine=0.4473, different_person=0.5382, impersonation=0.0145]`
- **Live Trial Mean**: `[genuine=0.402, different_person=0.566, impersonation=0.032]`
- **Consistency**: The predictions are strictly consistent. In all 10 live frames and the isolated diagnostic, C-Adv classifies the Native InSwapper artifact as `different_person` with a ~55% confidence, completely suppressing the `impersonation` class to near-zero. 

**Visual Inspection & Validity:**
- **Diagnostic Image**: `diagnostic_impersonation.jpg`
- **Validity**: **VALID TRIAL**. The image verifies a synthetic face-swap of the enrolled identity (Alice/Angelina Jolie) successfully rendered onto the physical participant.
- **Interpretation**: The pipeline correctly allowed the highly similar swapped face to pass the Model A gate. C-Adv reliably processed every frame but failed to classify the generative artifacts as synthetic `impersonation`. It classified them as `different_person`, securely dropping the final state to `UNKNOWN`.

---

## Summary Limitations
1. The genuine evaluation remains totally untested because the enrolled identity was not physically available for the camera feed.
2. Latency remains artificially high (~4.3s per frame during impersonation) due to the CPU execution constraints in the test environment.
