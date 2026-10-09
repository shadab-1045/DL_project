# Data Semantics Audit

## Genuine Semantics
- **Reference Identity**: Matches probe identity.
- **Probe Image**: A true unaltered image of the reference identity.
- **Leakage**: The reference gallery image is STRICTLY EXCLUDED from being used as a genuine probe. `leakage_detected` is FALSE across all splits.

## Different Person Semantics
- **Reference Identity**: e.g., Identity A.
- **Probe Image**: A true unaltered image of Identity B.
- **Diversity**: We fixed a combinatorial duplication error where the same single probe was reused 29 times for different person comparisons. The 870 training pairs now utilize 618 unique probe images, ensuring diverse identity mismatch representation.

## Impersonation Semantics
- **Reference Identity**: Identity A (the claimed identity, or source face).
- **Physical/Target Identity**: Identity B (the context body).
- **Probe Image**: A synthetic neural face swap where Identity A's face is presented on Identity B's body.
