# Final Demo Specification

## Purpose
Demonstrate identity recognition plus synthetic-impersonation detection in real time.

Use only consenting participants.

## Step 1 — Enrollment
Capture 5–10 reference images of a demo participant.

```text
camera → face alignment → ArcFace → gallery embedding
```

No training occurs during enrollment.

## Step 2 — Genuine user
Expected behavior:
```text
Identity: Alice
Identity similarity: high
Impersonation probability: low
Decision: VERIFIED
```

## Step 3 — Unknown user
Expected behavior:
```text
Identity similarity to Alice: low
Decision: UNKNOWN
```

## Step 4 — Face-swap impersonation
Another participant appears while Deep-Live-Cam presents Alice's face.

Pipeline:
```text
Bob webcam
  ↓
Deep-Live-Cam
  ↓
Alice-looking frame
  ↓
ArcFace → may resemble Alice
  ↓
Model C → synthetic presentation evidence
  ↓
Decision → SUSPECTED_IMPERSONATION
```

The exact result must be measured.

## Recommended UI
```text
┌────────────────────────┬───────────────────────────────┐
│       LIVE VIDEO       │ Identity: Alice              │
│                        │ Similarity: XX%               │
│                        │                              │
│                        │ Genuine: XX%                 │
│                        │ Impersonation: XX%            │
│                        │                              │
│                        │ ⚠ SUSPECTED IMPERSONATION    │
├────────────────────────┴───────────────────────────────┤
│ FPS: XX    Latency: XX ms    Window: XX frames         │
└─────────────────────────────────────────────────────────┘
```

## Comparison panel
Show:
```text
Model A — identity-only
Identity: Alice
Similarity: XX%

Model C — anti-impersonation
Identity: Alice
Impersonation evidence: XX%
Decision: SUSPECTED_IMPERSONATION
```

## Final presentation order
1. Enrollment.
2. Genuine verification.
3. Unknown-person test.
4. Activate face swap.
5. Show baseline identity recognition.
6. Show Model C flagging suspected impersonation.
7. Show measured offline robustness results.

## Offline fallback
Have a recorded, consented sequence ready if live GPU/integration fails. Replay it through the same detector/dashboard pipeline. Never fake a live result.

## Privacy
Use aliases. Keep demo data local. Do not publish participant imagery without permission.
