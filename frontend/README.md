# React Frontend for Adversarial Face Identity Project

This directory contains the lightweight React + Vite frontend for the live demonstration.

## Requirements
- Node.js (v16+)
- npm

## Setup & Installation

1. Navigate to the `frontend` directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

## Running the Application

1. **Start the backend server** (from the project root):
   ```bash
   # Ensure your Python virtual environment is activated
   uvicorn src.api.server:app --port 8000
   ```

2. **Start the Vite development server** (from the `frontend` directory):
   ```bash
   npm run dev
   ```

3. Open your browser to the local URL provided by Vite (typically `http://localhost:5173`).

## Interactive Demo Workflows

The frontend supports two major interactive workflows:

### 1. Identity Enrollment
Users can dynamically enroll a new identity directly from the browser:
- Supply an **Identity Name**.
- Select and upload **3-5 reference images**.
- The backend will extract embeddings, aggregate them, and set this identity as the active **SOURCE** identity.

### 2. Live Face-Swap (Synthetic Impersonation Test)
Users can test the system against a synthetic presentation attack:
- **Face Swap OFF**: Standard verification pipeline against the raw webcam feed.
- **Face Swap ON**: 
  - The backend runs `Native InsightFace InSwapper` to paste the enrolled **SOURCE** identity onto the live **TARGET** webcam feed.
  - The exact **POST-SWAP** frame is analyzed by the ML verification pipeline.
  - The frontend overlays the swapped frame on top of the local webcam feed, demonstrating exactly what the ML pipeline is evaluating.

## Architecture Details

- **No ML in Browser**: The frontend is strictly a UI layer. It captures webcam frames, encodes them as JPEGs, and sends them to the backend. It contains zero client-side ML models, face detectors, or inference logic.
- **WebSocket Protocol**: The browser connects to `ws://localhost:8000/ws/inference?condition={genuine|impersonation}` based on the Face Swap state.
  - Sends: Binary JPEG blob
  - Receives: JSON payload containing inference metrics and (if Face Swap is ON) the base64-encoded swapped image for rendering.
- **Frame Pacing**: The frontend uses a conservative, configurable capture interval (default 1.5s / ~0.6 FPS) to prevent swamping the CPU-bound backend. It also tracks in-flight requests and will skip captures if the backend is still processing the previous frame.

## User-Facing States

The UI renders the exact three states defined by the core project:
- **VERIFIED**: Identity gate accepted the face AND anti-impersonation model predicted genuine.
- **UNKNOWN**: Identity gate rejected the face OR anti-impersonation model did not establish a genuine identity. (Note: Does not explicitly mean "impersonation detected").
- **SUSPECTED_IMPERSONATION**: Anti-impersonation model explicitly classified the image as an impersonation attempt.
