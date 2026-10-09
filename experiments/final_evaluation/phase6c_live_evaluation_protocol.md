# Phase 6C: Live Evaluation Protocol

This document defines the strict, read-only protocol for executing and recording the final live validation of the pipeline. 

## Instrumentation Audit
The core inference engine (`src/runtime/live_pipeline.py`) inherently instruments every required variable without any modification. It dumps an exhaustive JSONL record for every processed frame. The logging already strictly tracks:
- `timestamp`
- `frame_id`
- `face_detected` (bool)
- `raw_model_a_similarity` (float)
- `model_a_raw_accept` (bool)
- `raw_c_probs` (list of 3 floats)
- `raw_c_class` (int)
- `final_state` (string)
- `total_latency` (float)

The only change made was a UI addition to `src/runtime/manual_live_validation.py` to allow operators to explicitly select the `[D] Different Person` condition, thereby keeping the output logs cleanly separated.

**Integrity Check**: 
- Inference logic was untouched.
- Phase 5 model hashes remain identically enforced at runtime.
- Model A threshold remains statically frozen at `0.244529`.

---

## Scenario A: Genuine Identity Condition

**Required Setup**:
1. The operator physically sitting in front of the webcam must be the **enrolled identity**.
2. The environment must be adequately lit.
3. The operator initiates the capture by selecting `[G]`.

**Execution**:
- **Number of frames**: 10 frames per session.
- **Process**: No synthetic face-swap is applied. The raw webcam feed flows directly into the alignment/ArcFace branch.
- **What is recorded**: A JSONL file `manual_genuine_<timestamp>.jsonl` containing the per-frame JSON outputs.

**Expected Observation**:
- `model_a_raw_accept` should be `True`.
- `raw_c_class` should be `0` (Genuine).
- `final_state` should be `VERIFIED`.

---

## Scenario B: Different-Person Condition

**Required Setup**:
1. The operator physically sitting in front of the webcam must **NOT** be the enrolled identity.
2. The operator initiates the capture by selecting `[D]`.

**Execution**:
- **Number of frames**: 10 frames per session.
- **Process**: No synthetic face-swap is applied. The raw webcam feed flows directly into the alignment/ArcFace branch.
- **What is recorded**: A JSONL file `manual_different_person_<timestamp>.jsonl`.

**Expected Observation**:
- `model_a_raw_accept` should be `False` (Model A rejects the identity outright).
- C-Adv is heavily bypassed/overridden by the decision engine mapping `False -> UNKNOWN`.
- `final_state` should be `UNKNOWN`.

---

## Scenario C: Native InSwapper Impersonation

**Required Setup**:
1. The operator physically sitting in front of the webcam must **NOT** be the enrolled identity.
2. The enrolled identity remains loaded as the "Source".
3. The operator initiates the capture by selecting `[I]`.

**Execution**:
- **Number of frames**: 10 frames per session.
- **Process**: The Native InSwapper intercepts the raw webcam feed and synthetically blends the source identity features onto the live operator's face. The post-swap frame flows into the evaluation branch.
- **What is recorded**: A JSONL file `manual_impersonation_<timestamp>.jsonl` and a static visual snapshot `diagnostic_impersonation.jpg` on the 5th frame for human verification.

**Expected Observation (based on prior diagnostics)**:
- `model_a_raw_accept` will likely be `True`.
- `raw_c_class` will likely be `1` (Different Person), as C-Adv generalizes poorly to this generator's artifacts in the live domain.
- `final_state` will resolve to `UNKNOWN`.

---

## Summarization and Conclusions
At the conclusion of the evaluations, the JSONL files will be aggregated.

**Observation vs Conclusion**:
- An **observation** is a direct statement of the recorded data (e.g., "C-Adv output `different_person` on 10/10 frames in Scenario C").
- A **conclusion** extrapolates the meaning of the data relative to the project goals (e.g., "The adversarially trained C-Adv model fails to recognize Native InSwapper artifacts as 'impersonation', meaning it successfully prevents false-verification but lacks the domain adaptation required to actively flag the attack"). 

**Known Limitations**:
- **Performance Constraints**: The tests run on the `CPUExecutionProvider` yielding ~0.2 FPS. This latency is artificially high due to hardware constraints but does not affect the mathematical integrity of the inference.
- **Fixed Model State**: This protocol strictly forbids retraining or adjusting thresholds based on live results, so the evaluation purely measures the baseline generalization gap.
