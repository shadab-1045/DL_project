# Final Decision Engine Logic Audit

This document formally specifies the implemented final decision logic, derived directly from `docs/PRD.md` and `docs/ARCHITECTURE.md`.

## Decision Pipeline

1. **Model A (Identity Engine)**
   Every incoming probe face is evaluated against the enrolled ArcFace gallery. 
   - If `similarity < threshold` ($\tau=0.2445$), the identity is not matched. The system immediately routes the decision to `UNKNOWN` because the presentation does not match the protected identity.
   - If `similarity >= threshold`, the identity is provisionally matched. The system proceeds to Model C to evaluate presentation authenticity.

2. **Model C (Anti-Impersonation Engine)**
   When Model A accepts the identity, Model C analyzes the probe and the claimed identity embedding, predicting one of three classes:
   - `genuine`
   - `different_person`
   - `impersonation`

3. **Final Decision Routing**
   The final semantic state is determined by the intersection of Model A and Model C:
   - **`VERIFIED`**: (Model A == MATCH) AND (Model C == `genuine`)
   - **`SUSPECTED_IMPERSONATION`**: (Model A == MATCH) AND (Model C == `impersonation`)
   - **`UNKNOWN`**: (Model A == NO MATCH) OR ((Model A == MATCH) AND (Model C == `different_person`))

## Interpretation
- **Model A** serves as the primary gate. If it rejects the identity, the system does not raise a security alarm (`SUSPECTED_IMPERSONATION`); it merely reports `UNKNOWN`. 
- **Model C** acts as a secondary filter. Even if Model A thinks the face matches, Model C can veto the decision and raise an alarm if it detects synthetic artifacts.
- The state `UNKNOWN` is the default fail-safe for both mismatched identities and ambiguous presentations (e.g., when Model C detects a different physical face but no synthetic swap).
