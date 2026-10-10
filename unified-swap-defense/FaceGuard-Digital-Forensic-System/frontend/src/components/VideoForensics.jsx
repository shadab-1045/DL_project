import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Film, ScanLine, AlertTriangle, CheckCircle, Play, Pause } from 'lucide-react';
import { ComposedChart, Bar, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine, ResponsiveContainer, Cell } from 'recharts';
import useFaceMesh from './useFaceMesh';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8001/faceguard';

const ChartTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const val = payload[0]?.value;
  return (
    <div style={{ background: 'rgba(6,12,20,0.96)', border: '1px solid rgba(0,229,255,0.2)', padding: '10px 14px', fontFamily: "'Share Tech Mono', monospace", fontSize: '0.72rem' }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 4 }}>T = {label}s</div>
      <div style={{ color: val > 0.4 ? '#ff2d6b' : '#00e5ff', fontWeight: 'bold' }}>RISK: {(val * 100).toFixed(1)}%</div>
      <div style={{ color: val > 0.4 ? '#ff2d6b' : 'var(--green)', fontSize: '0.6rem', marginTop: 4 }}>{val > 0.4 ? '⚠ ABOVE THRESHOLD' : '✓ BELOW THRESHOLD'}</div>
    </div>
  );
};

export default function VideoForensics({ threshold }) {
  const [file, setFile] = useState(null);
  const [videoURL, setVideoURL] = useState(null);
  const [result, setResult] = useState(null);
  const [scanning, setScanning] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrameSrc, setCurrentFrameSrc] = useState(null);
  const [currentScore, setCurrentScore] = useState(0);
  const [frameCount, setFrameCount] = useState(0);

  const leftVideoRef = useRef(null);
  const rightImgRef = useRef(null);
  const canvasRef = useRef(null);
  const sseRef = useRef(null);
  const { drawMeshOnCanvas } = useFaceMesh();

  const onDrop = useCallback((accepted) => {
    const f = accepted[0]; if (!f) return;
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    if (videoURL) { URL.revokeObjectURL(videoURL); }
    setFile(f); setVideoURL(URL.createObjectURL(f));
    setResult(null); setError(null); setProgress(0);
    setCurrentFrameSrc(null); setCurrentScore(0); setFrameCount(0); setIsPlaying(false);
  }, [videoURL]);

  // Cleanup object URLs to prevent memory leaks
  useEffect(() => {
    return () => {
      if (videoURL) { URL.revokeObjectURL(videoURL); }
    };
  }, [videoURL]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'video/*': ['.mp4', '.mov', '.avi', '.webm'] }, maxFiles: 1,
  });

  // Draw mesh whenever frame updates
  useEffect(() => {
    if (!currentFrameSrc) return;
    let active = true;
    const tryDraw = async () => {
      const img = rightImgRef.current; const canvas = canvasRef.current;
      if (!img || !canvas) return;
      if (!img.complete || img.naturalWidth === 0) {
        img.onload = async () => { if (active && canvas) { canvas.width = img.offsetWidth; canvas.height = img.offsetHeight; await drawMeshOnCanvas(img, canvas, { isFake: currentScore > threshold, score: currentScore, scanning: true }); } };
        return;
      }
      canvas.width = img.offsetWidth; canvas.height = img.offsetHeight;
      await drawMeshOnCanvas(img, canvas, { isFake: currentScore > threshold, score: currentScore, scanning: true });
    };
    tryDraw();
    return () => { active = false; };
  }, [currentFrameSrc, currentScore, threshold, drawMeshOnCanvas]);

  const togglePlay = () => {
    const lv = leftVideoRef.current; if (!lv) return;
    if (isPlaying) { lv.pause(); setIsPlaying(false); } else { lv.play(); setIsPlaying(true); }
  };

  const startScan = async () => {
    if (!file) return;
    setScanning(true); setError(null); setProgress(0); setResult(null);
    setCurrentFrameSrc(null); setCurrentScore(0); setFrameCount(0);
    try {
      const form = new FormData(); form.append('file', file);
      const uploadRes = await fetch(`${API}/upload/video`, { method: 'POST', body: form });
      if (!uploadRes.ok) throw new Error('Video upload failed');
      const { path } = await uploadRes.json();

      const evtSource = new EventSource(`${API}/analyze/video/stream?path=${encodeURIComponent(path)}`);
      sseRef.current = evtSource;

      evtSource.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        if (data.type === 'frame') {
          setProgress(data.progress);
          setCurrentFrameSrc(`data:image/jpeg;base64,${data.frame_b64}`);
          setCurrentScore(data.smoothed);
          setFrameCount(c => c + 1);
        }
        if (data.type === 'complete') {
          setResult(data); setProgress(100); setScanning(false); evtSource.close();
        }
        if (data.error) { setError(data.error); setScanning(false); evtSource.close(); }
      };
      evtSource.onerror = () => { setError('Stream lost. Is the backend running?'); setScanning(false); evtSource.close(); };
    } catch (e) { setError(e.message); setScanning(false); }
  };

  const clear = () => {
    if (sseRef.current) { sseRef.current.close(); sseRef.current = null; }
    setFile(null); setVideoURL(null); setResult(null); setScanning(false); setProgress(0);
    setCurrentFrameSrc(null); setCurrentScore(0); setFrameCount(0); setIsPlaying(false);
  };

  const isFake = result && result.average_score > threshold;
  const chartData = result?.timeline?.map(t => ({ time: t.time, score: t.score })) || [];
  const rightIsFake = currentScore > threshold;

  return (
    <div className="fade-in">
      <div className="panel">
        <div className="corner-tl" /><div className="corner-br" />
        <p className="panel-title">Video Forensic Analysis</p>

        {!file ? (
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'drag-active' : ''}`}>
            <input {...getInputProps()} />
            <Film className="dropzone-icon" />
            <div className="dropzone-text"><strong>Drop video evidence here</strong><br />or click to browse · MP4 · MOV · AVI · WEBM</div>
          </div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
              {/* LEFT — playable */}
              <div>
                <div style={{ fontSize: '0.6rem', fontFamily: "'Share Tech Mono'", color: 'var(--text-muted)', letterSpacing: 2, marginBottom: 6 }}>ORIGINAL EVIDENCE</div>
                <div style={{ position: 'relative', background: '#000', border: '1px solid var(--border)' }}>
                  <video ref={leftVideoRef} src={videoURL} style={{ width: '100%', display: 'block', maxHeight: 320 }} onEnded={() => setIsPlaying(false)} />
                  <button onClick={togglePlay} style={{ position: 'absolute', bottom: 10, left: '50%', transform: 'translateX(-50%)', background: 'rgba(0,229,255,0.1)', border: '1px solid var(--cyan)', color: 'var(--cyan)', padding: '6px 18px', cursor: 'pointer', fontFamily: "'Share Tech Mono'", fontSize: '0.7rem', display: 'flex', alignItems: 'center', gap: 6 }}>
                    {isPlaying ? <Pause size={12} /> : <Play size={12} />}{isPlaying ? 'PAUSE' : 'PLAY'}
                  </button>
                </div>
              </div>

              {/* RIGHT — frame-by-frame + real MediaPipe mesh */}
              <div>
                <div style={{ fontSize: '0.6rem', fontFamily: "'Share Tech Mono'", color: scanning ? 'var(--amber)' : result ? (isFake ? 'var(--magenta)' : 'var(--cyan)') : 'var(--text-muted)', letterSpacing: 2, marginBottom: 6, transition: 'color 0.3s' }}>
                  {scanning ? `⟳ ANALYZING FRAME ${frameCount} — ${progress}%` : result ? (isFake ? '⚠ THREAT DETECTED' : '✓ GENUINE CONTENT') : 'FORENSIC OUTPUT'}
                </div>
                <div style={{ position: 'relative', background: '#000', minHeight: 320, border: `1px solid ${scanning ? 'rgba(255,171,0,0.4)' : result ? (isFake ? 'rgba(255,45,107,0.55)' : 'rgba(0,229,255,0.4)') : 'var(--border)'}`, transition: 'border-color 0.4s', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  {currentFrameSrc ? (
                    <>
                      <img ref={rightImgRef} src={currentFrameSrc} alt="frame" crossOrigin="anonymous"
                        style={{ width: '100%', display: 'block', maxHeight: 320, objectFit: 'contain', filter: rightIsFake ? 'saturate(0.55) brightness(0.75)' : 'none', transition: 'filter 0.3s' }} />
                      <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
                    </>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontFamily: "'Share Tech Mono'", fontSize: '0.7rem', letterSpacing: 2 }}>
                      {scanning ? (<><div className="spinner-ring" style={{ margin: '0 auto 16px' }} /><div className="spinner-text">UPLOADING...</div></>) : 'AWAITING SCAN'}
                    </div>
                  )}
                  {scanning && (
                    <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0 }}>
                      <div className="progress-bar-wrap" style={{ height: 3, borderRadius: 0 }}>
                        <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            <div style={{ display: 'flex', gap: 24, marginBottom: 14 }}>
              <div className="stat-row" style={{ flex: 1 }}><span className="stat-key">FILE</span><span className="stat-val" style={{ fontSize: '0.65rem' }}>{file.name}</span></div>
              <div className="stat-row" style={{ flexShrink: 0 }}><span className="stat-key">SIZE</span><span className="stat-val">{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>
            </div>

            <div style={{ display: 'flex', gap: 12, marginBottom: 20 }}>
              <button className="btn" onClick={startScan} disabled={scanning}><ScanLine size={16} />{scanning ? `Scanning... ${progress}%` : 'Initiate Forensic Scan'}</button>
              <button className="btn" onClick={clear} disabled={scanning}>Clear</button>
            </div>

            {error && <div className="alert error">{error}</div>}

            {result && (
              <div className="fade-in">
                <div className={`verdict-block ${isFake ? 'fake' : 'real'}`} style={{ marginBottom: 24 }}>
                  <div>
                    <div className="verdict-label">Forensic Verdict</div>
                    <div className="verdict-text">{isFake ? <><AlertTriangle size={18} style={{ display: 'inline', marginRight: 8 }} />Manipulation Detected</> : <><CheckCircle size={18} style={{ display: 'inline', marginRight: 8 }} />Content Genuine</>}</div>
                    <div style={{ marginTop: 10, display: 'flex', gap: 24 }}>
                      {[['AVG RISK', `${(result.average_score * 100).toFixed(1)}%`], ['FRAMES', result.total_frames_analyzed], ['SUSPICIOUS', result.suspicious_frames.length]].map(([k, v]) => (
                        <div key={k}><div style={{ fontFamily: "'Share Tech Mono'", fontSize: '0.58rem', color: 'var(--text-muted)', letterSpacing: 2 }}>{k}</div><div style={{ fontFamily: "'Share Tech Mono'", fontSize: '0.9rem', color: 'var(--cyan)' }}>{v}</div></div>
                      ))}
                    </div>
                  </div>
                  <div className="verdict-score"><div className="score-value">{Math.round(result.average_score * 100)}</div><div className="score-label">RISK SCORE</div></div>
                </div>

                {chartData.length > 0 && (
                  <div style={{ background: 'var(--bg-deep)', border: '1px solid var(--border)', padding: '24px 20px', position: 'relative', marginBottom: 20 }}>
                    <div className="corner-tl" /><div className="corner-br" />
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 16 }}>
                      <div>
                        <div style={{ fontFamily: "'Exo 2'", fontWeight: 300, fontSize: '1rem', letterSpacing: 4, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>Temporal Risk Timeline</div>
                        <div style={{ fontFamily: "'Share Tech Mono'", fontSize: '0.62rem', color: 'var(--text-muted)', marginTop: 4 }}>{chartData.length} samples · threshold {(threshold * 100).toFixed(0)}%</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontFamily: "'Share Tech Mono'", fontSize: '0.6rem', color: 'var(--text-muted)' }}>AVG RISK</div>
                        <div style={{ fontFamily: "'Share Tech Mono'", fontSize: '1.4rem', color: isFake ? 'var(--magenta)' : 'var(--green)', lineHeight: 1 }}>{(result.average_score * 100).toFixed(1)}%</div>
                      </div>
                    </div>
                    <div style={{ display: 'flex', borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', marginBottom: 20 }}>
                      {[['FRAMES ANALYZED', chartData.length], ['SUSPICIOUS', result.suspicious_frames.length], ['PEAK RISK', `${(Math.max(...chartData.map(d => d.score)) * 100).toFixed(1)}%`], ['VERDICT', isFake ? '⚠ FAKE' : '✓ REAL']].map(([k, v], i) => (
                        <div key={k} style={{ flex: 1, padding: '10px 14px', borderRight: i < 3 ? '1px solid var(--border)' : 'none', textAlign: 'center' }}>
                          <div style={{ fontFamily: "'Share Tech Mono'", fontSize: '0.57rem', color: 'var(--text-muted)', letterSpacing: 2 }}>{k}</div>
                          <div style={{ fontFamily: "'Share Tech Mono'", fontSize: '0.95rem', marginTop: 4, color: k === 'VERDICT' ? (isFake ? 'var(--magenta)' : 'var(--green)') : k === 'SUSPICIOUS' && v > 0 ? 'var(--magenta)' : 'var(--cyan)' }}>{v}</div>
                        </div>
                      ))}
                    </div>
                    <ResponsiveContainer width="100%" height={260}>
                      <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: -10, bottom: 16 }}>
                        <defs>
                          <linearGradient id="bFake" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#ff2d6b" stopOpacity={0.9} /><stop offset="100%" stopColor="#ff2d6b" stopOpacity={0.3} /></linearGradient>
                          <linearGradient id="bSafe" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#00e5ff" stopOpacity={0.7} /><stop offset="100%" stopColor="#00e5ff" stopOpacity={0.15} /></linearGradient>
                        </defs>
                        <CartesianGrid stroke="rgba(0,229,255,0.05)" strokeDasharray="4 6" vertical={false} />
                        <XAxis dataKey="time" tickFormatter={v => `${v}s`} tick={{ fill: 'var(--text-muted)', fontFamily: "'Share Tech Mono'", fontSize: 10 }} axisLine={{ stroke: 'rgba(0,229,255,0.15)' }} tickLine={false} label={{ value: 'VIDEO TIMELINE (SECONDS)', position: 'insideBottom', offset: -8, fill: 'var(--text-muted)', fontFamily: "'Share Tech Mono'", fontSize: 9 }} />
                        <YAxis domain={[0, 1]} tickFormatter={v => `${Math.round(v * 100)}%`} tick={{ fill: 'var(--text-muted)', fontFamily: "'Share Tech Mono'", fontSize: 10 }} axisLine={false} tickLine={false} label={{ value: 'RISK LEVEL (%)', angle: -90, position: 'insideLeft', offset: 16, fill: 'var(--text-muted)', fontFamily: "'Share Tech Mono'", fontSize: 9 }} />
                        <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(0,229,255,0.04)' }} />
                        <ReferenceLine y={threshold} stroke="#ffab00" strokeDasharray="6 4" strokeWidth={1.5} label={{ value: `◆ DANGER ${(threshold * 100).toFixed(0)}%`, fill: '#ffab00', fontFamily: "'Share Tech Mono'", fontSize: 9, position: 'insideTopLeft' }} />
                        <Bar dataKey="score" maxBarSize={28} radius={[2, 2, 0, 0]}>{chartData.map((entry, i) => <Cell key={i} fill={entry.score > threshold ? 'url(#bFake)' : 'url(#bSafe)'} />)}</Bar>
                        <Line type="monotone" dataKey="score" stroke="#ffffff" strokeWidth={1.5} dot={{ fill: '#ff00ff', r: 3.5, strokeWidth: 0 }} activeDot={{ r: 5, fill: '#ff00ff', strokeWidth: 0 }} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {result.suspicious_frames.length > 0 && (
                  <div><div className="chart-title">SUSPICIOUS ARTIFACTS</div>
                    <div className="frames-grid">{result.suspicious_frames.map((f, i) => (<div key={i} className="frame-card fade-in"><img src={f.thumbnail} alt={`frame-${i}`} /><span className="frame-time">{f.time}s</span><span className="frame-score">{(f.score * 100).toFixed(0)}% RISK</span></div>))}</div>
                  </div>
                )}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

