import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Camera, CameraOff, Wifi, WifiOff } from 'lucide-react';
import RiskGauge from './RiskGauge';
import useFaceMesh from './useFaceMesh';

const API_WS = (process.env.REACT_APP_API_URL || 'http://localhost:8001/faceguard').replace(/^http/, 'ws');
const SEND_INTERVAL_MS = 200;

export default function WebcamScan({ threshold }) {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const wsRef       = useRef(null);
  const intervalRef = useRef(null);
  const streamRef   = useRef(null);
  const animRef     = useRef(null);
  const uptimeRef   = useRef(null);
  const scoreRef    = useRef(0);
  const isFakeRef   = useRef(false);

  const [running, setRunning]     = useState(false);
  const [connected, setConnected] = useState(false);
  const [score, setScore]         = useState(0);
  const [verdict, setVerdict]     = useState(null);
  const [error, setError]         = useState(null);
  const [uptime, setUptime]       = useState(0);

  const { drawMeshOnCanvas } = useFaceMesh();

  // HUD animation loop — draws on canvas at ~10fps
  useEffect(() => {
    if (!running) return;
    let active = true;

    const loop = async () => {
      if (!active) return;
      const video  = videoRef.current;
      const canvas = canvasRef.current;
      if (video && canvas && video.readyState >= 2) {
        canvas.width  = video.offsetWidth;
        canvas.height = video.offsetHeight;
        await drawMeshOnCanvas(video, canvas, {
          isFake:   isFakeRef.current,
          score:    scoreRef.current,
          scanning: false,
        });
      }
      animRef.current = setTimeout(() => { if (active) requestAnimationFrame(loop); }, 100);
    };

    requestAnimationFrame(loop);
    return () => { active = false; clearTimeout(animRef.current); };
  }, [running, drawMeshOnCanvas]);

  const stopCamera = useCallback(() => {
    clearInterval(intervalRef.current);
    clearInterval(uptimeRef.current);
    clearTimeout(animRef.current);
    if (wsRef.current)  { wsRef.current.close();  wsRef.current  = null; }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null; }
    setRunning(false); setConnected(false);
    setScore(0); setVerdict(null); setUptime(0);
    scoreRef.current = 0; isFakeRef.current = false;
  }, []);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      await videoRef.current.play();

      const ws = new WebSocket(`${API_WS}/ws/webcam`);
      wsRef.current = ws;
      ws.onopen  = () => setConnected(true);
      ws.onclose = () => setConnected(false);
      ws.onerror = () => setError('WebSocket failed. Is the backend running?');
      ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        if (data.smoothed_score !== undefined) {
          scoreRef.current  = data.smoothed_score;
          isFakeRef.current = data.smoothed_score > threshold;
          setScore(data.smoothed_score);
          setVerdict(data.verdict);
        }
      };

      // Send frames — draw to offscreen canvas without mirroring so backend sees real image
      const offscreen = document.createElement('canvas');
      offscreen.width = 320; offscreen.height = 240;
      const octx = offscreen.getContext('2d');
      intervalRef.current = setInterval(() => {
        if (ws.readyState !== WebSocket.OPEN || !videoRef.current) return;
        octx.drawImage(videoRef.current, 0, 0, 320, 240);
        ws.send(offscreen.toDataURL('image/jpeg', 0.7));
      }, SEND_INTERVAL_MS);

      uptimeRef.current = setInterval(() => setUptime(u => u + 1), 1000);
      setRunning(true);
    } catch (e) {
      setError(`Camera access denied: ${e.message}`);
    }
  }, [threshold]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const isFake = score > threshold;
  const fmt    = (n) => String(n).padStart(2, '0');

  return (
    <div className="fade-in">
      <div className="panel">
        <div className="corner-tl" /><div className="corner-br" />
        <p className="panel-title">Live Surveillance Feed</p>

        <div className="webcam-layout">
          {/* Video feed */}
          <div>
            <div style={{
              position: 'relative',
              background: '#000',
              border: `1px solid ${running && verdict
                ? (isFake ? 'rgba(255,45,107,0.55)' : 'rgba(0,212,255,0.4)')
                : 'var(--border)'}`,
              transition: 'border-color 0.4s',
              aspectRatio: '4/3',
              overflow: 'hidden',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              {/* Video — NOT mirrored. Natural camera output. */}
              <video
                ref={videoRef}
                muted
                playsInline
                style={{
                  display: running ? 'block' : 'none',
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  // No transform: scaleX(-1) — this was causing the mirror
                  filter: running && isFake ? 'saturate(0.6) brightness(0.78)' : 'none',
                  transition: 'filter 0.4s',
                }}
              />

              {/* HUD Canvas — no transform needed since video isn't flipped */}
              <canvas
                ref={canvasRef}
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                  display: running ? 'block' : 'none',
                }}
              />

              {!running && (
                <div className="webcam-inactive">
                  <CameraOff size={64} /><p>FEED OFFLINE</p>
                </div>
              )}

              {running && (
                <div style={{
                  position: 'absolute', bottom: 10, left: 12,
                  fontFamily: "'Share Tech Mono', monospace",
                  fontSize: '0.65rem',
                  color: 'rgba(0,212,255,0.7)',
                  letterSpacing: 1,
                }}>
                  REC ● {fmt(Math.floor(uptime / 60))}:{fmt(uptime % 60)}
                </div>
              )}
            </div>

            {/* Controls */}
            <div style={{ display: 'flex', gap: 12, marginTop: 14, alignItems: 'center' }}>
              {!running
                ? <button className="btn" onClick={startCamera}><Camera size={16} />Start Feed</button>
                : <button className="btn btn-danger" onClick={stopCamera}><CameraOff size={16} />Stop Feed</button>
              }
              <div style={{
                display: 'flex', alignItems: 'center', gap: 8,
                fontFamily: "'Share Tech Mono'", fontSize: '0.65rem',
                color: connected ? 'var(--green)' : 'var(--text-muted)',
              }}>
                {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
                {connected ? 'WS CONNECTED' : 'WS OFFLINE'}
              </div>
            </div>

            {error && <div className="alert warn" style={{ marginTop: 12 }}>{error}</div>}

            {running && (
              <div style={{ marginTop: 14, border: '1px solid var(--border)', background: 'var(--bg-deep)', padding: '12px 16px' }}>
                {[
                  ['STATUS',    'ACTIVE',                                          'var(--green)'],
                  ['UPTIME',    `${fmt(Math.floor(uptime/60))}:${fmt(uptime%60)}`],
                  ['INFERENCE', '~5 FPS'],
                  ['THRESHOLD', `${(threshold * 100).toFixed(0)}%`],
                ].map(([k, v, c]) => (
                  <div className="stat-row" key={k}>
                    <span className="stat-key">{k}</span>
                    <span className="stat-val" style={c ? { color: c } : {}}>{v}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Sidebar — gauge + verdict */}
          <div className="webcam-sidebar">
            <div style={{
              fontFamily: "'Share Tech Mono'", fontSize: '0.62rem',
              color: 'var(--text-muted)', letterSpacing: 3,
              textTransform: 'uppercase', marginBottom: 12,
            }}>
              Live Risk Monitor
            </div>

            <RiskGauge score={running ? score : 0} threshold={threshold} size={210} showZones />

            {running && verdict && (
              <div className={`verdict-block fade-in ${isFake ? 'fake' : 'real'}`} style={{ marginTop: 16 }}>
                <div>
                  <div className="verdict-label">Live Verdict</div>
                  <div className="verdict-text" style={{ fontSize: '1rem', letterSpacing: 3 }}>
                    {isFake ? '⚠ DEEPFAKE' : '✓ AUTHENTIC'}
                  </div>
                </div>
                <div className="verdict-score">
                  <div className="score-value" style={{ fontSize: '1.8rem' }}>{Math.round(score * 100)}</div>
                  <div className="score-label">RISK</div>
                </div>
              </div>
            )}

            {!running && (
              <div style={{
                textAlign: 'center', padding: '24px 0',
                color: 'var(--text-muted)',
                fontFamily: "'Share Tech Mono'", fontSize: '0.65rem', letterSpacing: 2,
              }}>
                START FEED TO<br />ACTIVATE MONITOR
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

