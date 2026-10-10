"""Full-stack smoke test: boots unified_api (real models) and sends real webcam-style frames over the WebSocket.
Needs models/swapdet_deployed.pt. Usage: python smoke_unified.py"""
import os, sys, cv2
from fastapi.testclient import TestClient
import unified_api

LIVE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "live_demo")
ROOT = os.path.dirname(os.path.abspath(__file__))
USER = os.path.join(ROOT, "eval_data", "demo_user_genuine.jpg")   # local only (gitignored): a normal-lighting webcam frame of the enrolled user
CASES = [  # (frame path, condition sent by the client, allowed verdicts)
    (os.path.join(LIVE, "manual/diagnostic_impersonation.jpg"), "genuine", {"ATTACK_BLOCKED"}),         # already-swapped webcam frame
    (os.path.join(LIVE, "manual/diagnostic_genuine.jpg"), "impersonation", {"ATTACK_BLOCKED"}),         # server applies the live swap
]
if os.path.exists(USER):
    CASES += [(USER, "genuine", {"SECURE_VERIFIED", "UNKNOWN_IDENTITY"}),                               # real face: must never be an attack
              (USER, "impersonation", {"ATTACK_BLOCKED"})]
STRESS = (os.path.join(LIVE, "manual/diagnostic_genuine.jpg"), "genuine")  # blown-out backlight + overlay: known false-block, informational

def verdicts(client, path, condition, n=4):
    img = cv2.imread(path); ok, buf = cv2.imencode(".jpg", img); out = []
    with client.websocket_connect(f"/ws/unified-verify?condition={condition}") as ws:
        for _ in range(n):
            ws.send_bytes(buf.tobytes()); out.append(ws.receive_json())
    return out

if __name__ == "__main__":
    failed = 0
    with TestClient(unified_api.app) as client:
        for path, cond, allowed in CASES:
            r = verdicts(client, path, cond)[-1]
            ok = r.get("overall_verdict") in allowed; failed += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {os.path.basename(path)} [{cond}] -> {r.get('overall_verdict')}  P(swap)={r.get('swap_probability')}  id={r.get('identity')}  | {r.get('reason')}")
        r = verdicts(client, *STRESS)[-1]
        print(f"INFO  stress (blown-out lighting) {os.path.basename(STRESS[0])} -> {r.get('overall_verdict')} P(swap)={r.get('swap_probability'):.2f}  [documented limitation, not gated]")
    if not os.path.exists(USER): print("NOTE  eval_data/demo_user_genuine.jpg missing: genuine-user case skipped")
    sys.exit(1 if failed else 0)
