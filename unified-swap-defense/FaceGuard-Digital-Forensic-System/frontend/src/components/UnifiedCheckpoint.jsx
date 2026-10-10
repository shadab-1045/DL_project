import React, { useState, useEffect, useRef } from 'react';
import { Shield, ShieldAlert, ShieldCheck, AlertTriangle, Camera, Play, Square, Swords, ScanFace } from 'lucide-react';
import './UnifiedCheckpoint.css';

const WS_URL = 'ws://localhost:8001/ws/unified-verify';

// verdict -> presentation. Keys are the server's overall_verdict values.
const STATES = {
  SECURE_VERIFIED: { tone: 'ok',   label: 'VERIFIED', title: 'Identity verified',        Icon: ShieldCheck },
  ATTACK_BLOCKED:  { tone: 'bad',  label: 'BLOCKED',  title: 'Face-swap attack blocked', Icon: ShieldAlert },
  UNKNOWN_IDENTITY:{ tone: 'warn', label: 'UNKNOWN',  title: 'Real face, not enrolled',  Icon: AlertTriangle },
  NO_FACE:         { tone: 'idle', label: 'NO FACE',  title: 'No face in frame',         Icon: ScanFace },
  ANALYZING:       { tone: 'idle', label: 'ANALYZING', title: 'Collecting evidence...',  Icon: Shield },
};
const WAITING = { tone: 'idle', label: 'WAITING', title: 'Waiting for first inference...', Icon: Shield };

// horizontal meter for a 0..1 value with an optional decision-threshold marker
function Meter({ value, threshold, tone }) {
  const pct = (v) => `${Math.max(0, Math.min(1, v)) * 100}%`;
  return (
    <div className="uc-meter" role="meter" aria-valuemin={0} aria-valuemax={1} aria-valuenow={value ?? 0}>
      <div className={`uc-meter-fill uc-${tone}`} style={{ transform: `scaleX(${value == null ? 0 : Math.max(0, Math.min(1, value))})` }} />
      {threshold != null && <div className="uc-meter-thr" style={{ left: pct(threshold) }} title={`threshold ${threshold.toFixed(2)}`} />}
    </div>
  );
}

const Row = ({ k, children }) => (
  <div className="uc-row"><span className="uc-k">{k}</span><span className="uc-v">{children}</span></div>
);
// G / D / I class mix derived from the two real signals (NOT a trained 3-class head).
//  I: swap score rescaled piecewise-linearly so the block threshold maps to 0.5 (I >= 0.5 <=> blocked).
//  G / D: the remaining mass split by a soft ArcFace accept test (sigmoid of similarity - threshold).
function classMix(info) {
  if (!info || info.swap_probability == null) return null;
  const p = info.swap_probability, t = info.swap_threshold;
  const I = p < t ? 0.5 * p / t : 0.5 + 0.5 * (p - t) / (1 - t);
  const known = 1 / (1 + Math.exp(-(info.identity_similarity - info.identity_threshold) / 0.05));
  return { G: (1 - I) * known, D: (1 - I) * (1 - known), I };
}
const fmt = (x, d = 3) => (x == null || Number.isNaN(x) ? 'N/A' : Number(x).toFixed(d));

export default function UnifiedCheckpoint() {
  const [active, setActive] = useState(false);
  const [status, setStatus] = useState('Idle');
  const [info, setInfo] = useState(null);       // last server message
  const [swapImg, setSwapImg] = useState(null); // frame the verifier sees (swapped) when the attack is simulated
  const [attack, setAttack] = useState(false);  // simulate a live InSwapper impersonation attack

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);

  const reset = () => { setInfo(null); setSwapImg(null); };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      if (videoRef.current) { videoRef.current.srcObject = stream; videoRef.current.play(); }
      streamRef.current = stream;
      connectWS(attack);
      setActive(true);
    } catch (err) {
      console.error(err);
      setStatus('Camera error - allow camera access and close other apps using it');
    }
  };

  const stopCamera = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
    if (wsRef.current) wsRef.current.close();
    setActive(false);
    setStatus('Idle');
    reset();
  };

  const connectWS = (simulateAttack) => {
    const ws = new WebSocket(`${WS_URL}?condition=${simulateAttack ? 'impersonation' : 'genuine'}`);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => { setStatus('Connected'); captureLoop(); };  // start only once the socket is OPEN
    ws.onclose = () => { reset(); setStatus('Disconnected'); };
    ws.onerror = () => setStatus('Connection error - is the backend running on :8001?');
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.error) { setStatus(data.error); reset(); return; }  // never leave a stale VERIFIED on screen
      setInfo(data);
      setSwapImg(data.processed_frame ? `data:image/jpeg;base64,${data.processed_frame}` : null);
    };
    wsRef.current = ws;
  };

  const captureLoop = () => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const canvas = canvasRef.current, video = videoRef.current;
    if (canvas && video && video.readyState === 4) {
      canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob((blob) => {
        if (blob && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) wsRef.current.send(blob);
      }, 'image/jpeg', 0.8);
    }
    setTimeout(captureLoop, 200); // 5 FPS
  };

  useEffect(() => () => stopCamera(), []); // eslint-disable-line react-hooks/exhaustive-deps

  const st = (info && STATES[info.overall_verdict]) || WAITING;
  const swapP = info ? info.swap_probability : null;
  const idSim = info ? info.identity_similarity : null;
  const waiting = !info;
  const swapBlocked = swapP != null && info && swapP >= info.swap_threshold;
  const mix = classMix(info);
  const mixClass = mix ? (mix.I >= 0.5 ? 'IMPERSONATION' : mix.G >= mix.D ? 'GENUINE' : 'DIFFERENT PERSON') : null;

  return (
    <div className="uc-page">
      <header className="uc-head">
        <h1>Unified Security Checkpoint</h1>
        <p>ArcFace identity matching plus a face-swap detector that can veto an identity match.</p>
        <p className="uc-threat">Threat model: live face-swap presentation attacks. The detector is not hardened against pixel-level adversarial perturbations (see README).</p>
      </header>

      <div className="uc-grid">
        {/* ---- left: camera + controls ---- */}
        <section className="uc-card">
          <div className="uc-cam">
            <video ref={videoRef} autoPlay playsInline muted />
            {!active && <div className="uc-cam-empty"><Camera size={40} /><span>Camera is off</span></div>}
            <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
          </div>

          <div className="uc-controls">
            <div className="uc-seg" role="group" aria-label="Scenario">
              <button className={!attack ? 'on' : ''} disabled={active} onClick={() => setAttack(false)}><ShieldCheck size={15} /> Genuine user</button>
              <button className={attack ? 'on bad' : ''} disabled={active} onClick={() => setAttack(true)}><Swords size={15} /> Simulate swap attack</button>
            </div>
            <button className={`uc-go ${active ? 'stop' : ''}`} onClick={active ? stopCamera : startCamera}>
              {active ? <><Square size={15} /> Stop session</> : <><Play size={15} /> Start verification</>}
            </button>
          </div>
          <div className="uc-status" aria-live="polite">Status: {status}{active && ` - ${attack ? 'attack simulation ON' : 'genuine'}`}</div>

          {swapImg && (
            <div className="uc-verifier">
              <div className="uc-verifier-label">WHAT THE VERIFIER SEES - your face swapped to the enrolled victim</div>
              <img src={swapImg} alt="Swapped frame seen by the verifier" />
            </div>
          )}
          {!active && (
            <ol className="uc-steps">
              <li>Run with <b>Genuine user</b>: you should be <b>VERIFIED</b> (enroll first: <code>python enroll_user.py Arsal --webcam</code>).</li>
              <li>Stop, choose <b>Simulate swap attack</b>, start again: ArcFace now sees the victim, the detector should <b>BLOCK</b> it.</li>
            </ol>
          )}
        </section>

        {/* ---- right: verdict + inference result ---- */}
        <section className="uc-side">
          <div className={`uc-verdict uc-${st.tone}`} aria-live="polite">
            <st.Icon size={44} />
            <div>
              <div className="uc-verdict-label">{st.label}</div>
              <div className="uc-verdict-title">{st.title}</div>
            </div>
          </div>
          {info && info.reason && <p className="uc-reason">{info.reason}</p>}

          <div className="uc-card uc-result">
            <h2>Inference Result</h2>
            <Row k="FINAL STATE"><span className={`uc-chip uc-${st.tone}`}>{st.label}</span></Row>
            <Row k="Face detection">{waiting ? 'N/A' : info.face_detected ? 'Detected' : 'Not detected'}</Row>

            <h3>Model A - identity (ArcFace)</h3>
            <Row k="Matched identity">{info && info.identity ? info.identity : 'no match'}</Row>
            <Row k="Similarity">{fmt(idSim)}</Row>
            <Row k="Threshold">{fmt(info && info.identity_threshold)}</Row>
            <Row k="Accepted"><b className={waiting ? '' : info.identity_accepted_by_arcface ? 'uc-t-ok' : 'uc-t-bad'}>{waiting ? 'N/A' : info.identity_accepted_by_arcface ? 'YES' : 'NO'}</b></Row>
            <Meter value={idSim} threshold={info && info.identity_threshold} tone={info && info.identity_accepted_by_arcface ? 'ok' : 'warn'} />

            <h3>Swap detector (MobileNetV2)</h3>
            <Row k="P(swap)">{swapP == null ? 'N/A' : `${(swapP * 100).toFixed(1)}%`}</Row>
            <Row k="Block threshold">{info ? `${(info.swap_threshold * 100).toFixed(0)}%` : 'N/A'}</Row>
            <Row k="Class"><b className={swapP == null ? '' : swapBlocked ? 'uc-t-bad' : 'uc-t-ok'}>{swapP == null ? 'N/A' : swapBlocked ? 'SYNTHETIC' : 'REAL'}</b></Row>
            <Meter value={swapP} threshold={info && info.swap_threshold} tone={swapBlocked ? 'bad' : 'ok'} />

            <h3>Anti-impersonation class (derived)</h3>
            <Row k="Class"><b className={mixClass === 'GENUINE' ? 'uc-t-ok' : mixClass ? 'uc-t-bad' : ''}>{mixClass ?? 'N/A'}</b></Row>
            <Row k="G (genuine)">{fmt(mix && mix.G)}</Row>
            <Row k="D (different person)">{fmt(mix && mix.D)}</Row>
            <Row k="I (impersonation)">{fmt(mix && mix.I)}</Row>
            <p className="uc-note">Derived from ArcFace similarity and the swap score; not a separate trained model. I &ge; 0.5 means the attempt is blocked.</p>

            <h3>Pipeline</h3>
            <Row k="Latency">{info ? `${info.latency_ms} ms` : 'N/A'}</Row>
            <Row k="Frame">{info ? info.frame_id : 'N/A'}</Row>
            <Row k="Swap applied">{info ? (info.swap_applied ? 'yes (live InSwapper)' : 'no') : 'N/A'}</Row>
          </div>
        </section>
      </div>
    </div>
  );
}
