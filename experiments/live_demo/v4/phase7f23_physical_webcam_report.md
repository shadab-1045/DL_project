# Phase 7F.23: Controlled Physical Webcam Validation

**STATUS: BLOCKED (HARDWARE & SUBJECTS UNAVAILABLE)**

As an autonomous AI agent running in a headless execution environment, I do not have access to a physical webcam, nor can I provide consenting human subjects. In strict adherence to the Phase 7F.23 protocol, I am halting execution of this phase. I have not simulated, faked, or substituted static images for this validation.

To complete this phase, a human operator must manually perform the validation using the provided checklist below.

---

## Operator-Run Checklist for Physical Webcam Validation

### Pre-Flight Integrity Check
- [ ] Verify `experiments/model_c_v4/best_fusion_model.pt` hash is exactly `f2647012bc2863d988e65f980310cac9b60a0beb8c3e068f38b564f4383be2f5`.
- [ ] Verify Model A threshold in `src/api/server.py` (via `LiveInferencePipelineV4`) remains strictly `0.244529`.
- [ ] Ensure the Phase 7F.20 JPEG parity adapter remains active in the pipeline.
- [ ] Ensure 2 consenting human participants are physically present.

### Condition 1: Enrolled Genuine Subject
- [ ] Have Participant A (Enrolled Identity) enroll in the system using 3-5 reference images via the webcam.
- [ ] Have Participant A sit in front of the live webcam.
- [ ] Record at least 10 consecutive frames.
- [ ] Verify Model A similarity > `0.244529` (Accepts).
- [ ] Verify V4 correctly extracts `P(Id_Match)` and `P(Synthetic)`.
- [ ] Log the final UI State (Expected: `VERIFIED`). Note any OOD False Positives (e.g., `SUSPECTED_IMPERSONATION`).

### Condition 2: Different Person
- [ ] Keep Participant A as the enrolled identity.
- [ ] Have Participant B sit in front of the live webcam.
- [ ] Record at least 10 consecutive frames.
- [ ] Verify Model A securely rejects the face (`Similarity < 0.244529`).
- [ ] Verify V4 is correctly bypassed to save compute.
- [ ] Log the final UI State (Expected: `UNKNOWN`).

### Condition 3: Native InSwapper Impersonation
- [ ] Have Participant B sit in front of the webcam.
- [ ] Enable the `Native InSwapper` attack, targeting Participant A (Enrolled Identity) onto Participant B's live feed.
- [ ] Record at least 10 consecutive post-swap frames.
- [ ] Verify Model A is fooled (`Similarity > 0.244529`).
- [ ] Verify V4 successfully detects the deepfake artifacts (`P(Synthetic) ~ 1.0`).
- [ ] Log the final UI State (Expected: `SUSPECTED_IMPERSONATION`).

### Reporting Guidelines
- Record all failures, missing detections, and rejections. Do not silently exclude them.
- Save telemetry logs to `experiments/live_demo/v4/physical/`.
- Do not claim statistical robustness from a single 10-frame session. This is qualitative deployment validation only.
