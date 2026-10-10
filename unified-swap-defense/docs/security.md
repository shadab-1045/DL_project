# Security and observability

This page covers the **API** and its hardening, and then the model-level threat model. The API is built for local demo use, not for exposure to a network.

> [!IMPORTANT]
> The API has **no authentication or authorization**. Bind it to `127.0.0.1` only. Do not expose it publicly.

## API hardening

What the code does today:

| Control | Where | Behaviour |
|---|---|---|
| Origin check (`OriginGuard`) | `unified_api.py` | For WebSocket upgrades and non-`GET/HEAD/OPTIONS` requests that carry an `Origin` header not in `ALLOWED_ORIGINS`: REST gets `403`, WebSockets are closed with code `1008`. Stops a hostile web page from opening the socket or POSTing to `/dl/enroll` from a victim's browser. |
| CORS allow-list | `unified_api.py`, `backend/main.py` | Same `ALLOWED_ORIGINS`: `localhost` and `127.0.0.1` on ports 3000 and 3005 by default; override with the `ALLOWED_ORIGINS` environment variable. |
| Frame size cap | `unified_api.py` | Frames over 2 MB on `/ws/unified-verify` are dropped. |
| Decode pixel cap | `unified_api.py` | `OPENCV_IO_MAX_IMAGE_PIXELS` set to 25,000,000 before OpenCV loads, to refuse decompression bombs. |
| Upload caps | `backend/main.py` | 10 MB images, 200 MB videos, read in a bounded way. |
| Opaque upload ids | `backend/main.py` | `/upload/video` stores the file under a random id; `/analyze/video/stream` accepts only that id and pops it, never a client path. Uploads older than an hour are deleted lazily, on the next upload. |
| Fail closed | `unified_api.py` | On an internal error the socket gets an error message instead of a stale "verified". |

### Known gaps

- **Requests without an `Origin` header are not checked.** `curl` and server-side clients pass straight through. `OriginGuard` protects against browsers, not against a direct client. Without authentication, anything that can reach the port can use the API.
- **No rate limiting** and no limit on concurrent WebSockets.
- **Decode cap is global, not per-endpoint.** The 25 MP limit is an environment setting; there is no explicit resolution check in the verdict handler.
- **A shared lock serialises the identity/swap step** (`pipeline_lock`). One slow client delays others.
- **Enrollment is open.** Anyone who can reach `/dl/enroll` can enroll a face as an identity.

## Observability

Both ends are instrumented with OpenTelemetry, but **only to the console**.

| Side | Setup | Output |
|---|---|---|
| Backend | `TracerProvider` + `BatchSpanProcessor(ConsoleSpanExporter)`, passed to FastAPI via its `telemetry=` argument | Spans printed to the server's stdout |
| Frontend | `WebTracerProvider` + `SimpleSpanProcessor(ConsoleSpanExporter)` with web auto-instrumentations; W3C trace-context headers are added to all fetches (`propagateTraceHeaderCorsUrls: /.*/`) | Spans printed to the browser console |

No collector, exporter or dashboard is configured, so traces are not stored or correlated across the two sides unless you read both consoles. Adding an OTLP exporter is the missing step.

## Threat model for the detector

| Attack | Defended? | Evidence |
|---|---|---|
| Live InSwapper swap shown to a camera, no knowledge of the detector | Largely, on the test set | 96.2% caught at the deployed threshold ([Evaluation](evaluation.md)) |
| Other swap tools (Deep-Live-Cam, SimSwap, diffusion-based) | **Unknown** | The detector saw only InSwapper output. Untested. |
| White-box perturbation of the face crop | **No** | PGD-10 at ε = 1/255 drops detection to 0% |
| Replay of a recorded genuine video, or a printed photo | **No** | No liveness check exists |
| Poisoned enrollment | **No** | Enrollment is unauthenticated |
| Attack on ArcFace itself | **Not evaluated** | The white-box tests only attack the detector |

## Reporting problems

This is coursework. If you find a security issue, open an issue on the repository.
