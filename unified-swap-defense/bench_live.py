"""Latency of the live attack path through the real WebSocket handler. Usage: python bench_live.py"""
import os, time, cv2, numpy as np
import torch  # noqa: F401
from fastapi.testclient import TestClient
import unified_api
from src.api import server as dl_server  # noqa: E402  (unified_api put the DL_project root on sys.path)

frame = cv2.imread(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments", "live_demo", "manual", "diagnostic_genuine.jpg"))
buf = cv2.imencode(".jpg", frame)[1].tobytes()
with TestClient(unified_api.app) as c:
    p = dl_server.pipeline_instance
    print("swapper providers :", p.swapper.swapper.session.get_providers())
    print("detector providers:", [m.session.get_providers()[0] for m in p.identity_pipeline.engine.app.models.values()][:2])
    for cond in ("genuine", "impersonation"):
        with c.websocket_connect(f"/ws/unified-verify?condition={cond}") as ws:
            for _ in range(3): ws.send_bytes(buf); ws.receive_json()          # warm-up
            t = []
            for _ in range(25):
                t0 = time.time(); ws.send_bytes(buf); r = ws.receive_json(); t.append(time.time() - t0)
        lat = r["dl_project_result"]
        print(f"{cond:14s} {1 / np.mean(t):5.1f} FPS  ({np.mean(t) * 1000:.0f} ms/frame)  swap={lat.get('latency_swap', lat.get('swap_latency', 0)) * 1000:.0f}ms detect={lat.get('latency_detect_align', 0) * 1000:.0f}ms arcface={lat.get('latency_arcface', lat.get('model_a_latency', 0)) * 1000:.0f}ms  verdict={r['overall_verdict']}")
