# Human Visual QA Summary

## Overview
This document summarizes the results of a direct human visual inspection of 10 randomly selected face-swap samples generated during Phase 2. This inspection serves as the primary ground-truth verification of swap directionality and quality, replacing automated cosine similarity checks which serve only as secondary diagnostics.

## Inspection Results
- **Samples Inspected**: 10 (4 Train, 3 Validation, 3 Test)
- **Number Direction-Correct**: 10
- **Number Direction-Incorrect**: 0
- **Number Unacceptable-Quality**: 0

## Exactly Inspected Sample IDs and Sheet Mapping
1. `qa_sheet_train_0.jpg` -> `SW_61fae7ff` (Train)
2. `qa_sheet_train_1.jpg` -> `SW_1fdac72c` (Train)
3. `qa_sheet_train_2.jpg` -> `SW_5e20059f` (Train)
4. `qa_sheet_train_3.jpg` -> `SW_2d706a5e` (Train)
5. `qa_sheet_val_4.jpg` -> `SW_e7df4ded` (Val)
6. `qa_sheet_val_5.jpg` -> `SW_240982e0` (Val)
7. `qa_sheet_val_6.jpg` -> `SW_27ca212f` (Val)
8. `qa_sheet_test_7.jpg` -> `SW_df366c81` (Test)
9. `qa_sheet_test_8.jpg` -> `SW_a1dde1af` (Test)
10. `qa_sheet_test_9.jpg` -> `SW_28ae05bb` (Test)

## Findings
The `inswapper_128.onnx` model correctly replaced the Target (physical) identity's face with the Source (reference) identity's face across all evaluated examples. The context, body, clothing, and background of the Target were consistently preserved. The resulting generated identity successfully aligns with the specified Source identity (the `visible_identity`). 

**Observed Artifacts (Isolated, Non-Systemic)**:
1. **Glasses Removal**: In `SW_e7df4ded` and `SW_240982e0` (Validation), the original target wore glasses/sunglasses. The generator successfully swapped the face but stripped or partially ghosted the glasses. The swap remains semantically valid and direction-correct.
2. **Skin Tone / Boundary Blending**: In `SW_1fdac72c` (Train), a female source was mapped onto a male target with a distinct haircut. The face replacement was correct, though minor discoloration boundaries were visible near the hairline. This is expected behavior for 128x128 resolution generic models and does not invalidate the impersonation label.

No catastrophic failures, missing faces, or reversed-direction swaps were found. The failures are strictly isolated image-level artifacts, entirely acceptable for the purpose of adversarial robust-feature training.

## Conclusion
The generation pipeline has successfully modeled and executed the semantic relationship required:
`Source Face + Target Context = Impersonation Presentation of Source Identity`

The data is fully approved for Phase 3.
