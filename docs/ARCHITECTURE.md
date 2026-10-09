# Technical Architecture

## 1. High-level

```text
WEBCAM
  ↓
Deep-Live-Cam
  ↓
processed frame
  ↓
face detection + alignment
  ↓
┌───────────────────────┬────────────────────────┐
│ Identity branch       │ Anti-impersonation     │
│ InsightFace/ArcFace   │ Model C                │
└──────────┬────────────┴─────────────┬──────────┘
           ↓                          ↓
      identity score          impersonation score
           └──────────────┬───────────────┘
                          ↓
                   decision engine
                          ↓
                temporal smoothing
                          ↓
                    WebSocket
                          ↓
                      React UI
```

## 2. Responsibilities

### Deep-Live-Cam
- webcam interaction
- face-swap generation
- processed frame output

It is not the anti-impersonation model.

### Preprocessing
- face detection
- alignment
- normal face crop
- expanded/context crop

### Identity engine
- ArcFace embedding extraction
- gallery matching
- identity similarity

### Model C
- visual feature extraction
- fuse visual features with claimed-identity ArcFace embedding
- classify `genuine`, `different_person`, `impersonation`

### Decision engine
Combine identity similarity, anti-impersonation output, and temporal state.

### Backend
FastAPI for session/enrollment/inference APIs and WebSocket result transport.

### Frontend
React for enrollment, live display, metrics, and decision state.

## 3. Model interfaces

### Model A
Input: aligned face.
Output:
```json
{
  "identity": "P001",
  "similarity": 0.92
}
```

### Model C
Inputs:
- face/context tensor
- normalized claimed-identity ArcFace embedding

Output:
```json
{
  "genuine": 0.0,
  "different_person": 0.0,
  "impersonation": 0.0
}
```

## 4. Runtime object

```json
{
  "frame_id": 123,
  "identity": "P001",
  "identity_similarity": 0.92,
  "genuine": 0.04,
  "different_person": 0.02,
  "impersonation": 0.94,
  "decision": "SUSPECTED_IMPERSONATION",
  "fps": 24.0
}
```

All numbers are examples, not expected results.

## 5. Integration rule
Prefer in-memory frame transfer from Deep-Live-Cam into the security pipeline. Avoid screen capture and per-frame disk I/O.

## 6. Failure handling
- no face → `NO_FACE`
- low-quality face → `LOW_QUALITY`
- no identity above threshold → `UNKNOWN`
- model failure → error/fail-safe, never `VERIFIED`
- multiple faces → defined selection/multi-face policy
- WebSocket failure → visible connection state

## 7. Enrollment
Reference images → face alignment → ArcFace embeddings → normalized gallery representation.

Enrollment does not retrain any model.

## 8. Temporal processing
Maintain a configurable recent prediction window, e.g. 10–15 frames. Aggregate anti-impersonation evidence using a documented method and expose the window size in configuration.
