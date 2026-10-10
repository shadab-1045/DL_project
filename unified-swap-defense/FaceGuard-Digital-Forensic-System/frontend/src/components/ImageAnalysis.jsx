import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, ScanLine, AlertTriangle, CheckCircle } from 'lucide-react';
import RiskGauge from './RiskGauge';
import useFaceMesh from './useFaceMesh';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8001/faceguard';

export default function ImageAnalysis({ threshold }) {
  const [file, setFile]       = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult]   = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(null);

  const rightImgRef = useRef(null);
  const canvasRef   = useRef(null);
  const animRef     = useRef(null);

  const { drawMeshOnCanvas, ready } = useFaceMesh();

  const onDrop = useCallback((accepted) => {
    const f = accepted[0]; if (!f) return;
    setFile(f); setPreview(URL.createObjectURL(f)); setResult(null); setError(null);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] }, maxFiles: 1,
  });

  // Animate real MediaPipe mesh on right image
  useEffect(() => {
    if (!preview) { cancelAnimationFrame(animRef.current); return; }
    let active = true;

    const loop = async () => {
      if (!active) return;
      const img = rightImgRef.current; const canvas = canvasRef.current;
      if (!img || !canvas || !img.complete || img.naturalWidth === 0) {
        if (active) animRef.current = requestAnimationFrame(loop);
        return;
      }
      canvas.width  = img.offsetWidth;
      canvas.height = img.offsetHeight;
      const isFake = result ? result.score > threshold : false;
      await drawMeshOnCanvas(img, canvas, { isFake, score: result?.score || 0, scanning: !result });
      if (active) animRef.current = setTimeout(() => requestAnimationFrame(loop), 66);
    };

    requestAnimationFrame(loop);
    return () => { active = false; cancelAnimationFrame(animRef.current); clearTimeout(animRef.current); };
  }, [preview, result, threshold, drawMeshOnCanvas]);

  const analyze = async () => {
    if (!file) return;
    setLoading(true); setError(null);
    try {
      const form = new FormData(); form.append('file', file);
      const res = await fetch(`${API}/analyze/image`, { method: 'POST', body: form });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      setResult(await res.json());
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const clear = () => {
    cancelAnimationFrame(animRef.current); clearTimeout(animRef.current);
    setFile(null); setPreview(null); setResult(null); setError(null);
  };

  const isFake = result && result.score > threshold;

  return (
    <div className="fade-in">
      <div className="panel">
        <div className="corner-tl" /><div className="corner-br" />
        <p className="panel-title">Image Forensic Analysis</p>

        {!preview ? (
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'drag-active' : ''}`}>
            <input {...getInputProps()} />
            <Upload className="dropzone-icon" />
            <div className="dropzone-text">
              <strong>Drop image evidence here</strong>
              <br />or click to browse · JPG · PNG · WEBP
              {!ready && (
                <span style={{ display: 'block', color: 'var(--amber)', marginTop: 6, fontSize: '0.65rem' }}>
                  ⟳ Loading MediaPipe face detection...
                </span>
              )}
            </div>
          </div>
        ) : (
          <>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 20 }}>
              {/* LEFT — clean original */}
              <div style={{ position: 'relative', background: '#000', border: '1px solid var(--border)' }}>
                <span className="preview-box-label">ORIGINAL EVIDENCE</span>
                <img src={preview} alt="original"
                  style={{ width: '100%', display: 'block', maxHeight: 460, objectFit: 'contain' }} />
              </div>

              {/* RIGHT — real MediaPipe mesh overlay */}
              <div style={{
                position: 'relative', background: '#000',
                border: `1px solid ${result ? (isFake ? 'rgba(255,45,107,0.55)' : 'rgba(0,229,255,0.4)') : 'var(--border)'}`,
                transition: 'border-color 0.4s',
              }}>
                <span className="preview-box-label" style={{ color: result ? (isFake ? 'var(--magenta)' : 'var(--cyan)') : 'var(--text-muted)' }}>
                  {loading ? '⟳ SCANNING...' : result ? (isFake ? '⚠ THREAT DETECTED' : '✓ AUTHENTIC') : 'FORENSIC OUTPUT'}
                </span>
                <img ref={rightImgRef} src={preview} alt="scan" crossOrigin="anonymous"
                  style={{
                    width: '100%', display: 'block', maxHeight: 460, objectFit: 'contain',
                    filter: result && isFake ? 'saturate(0.6) brightness(0.8)' : 'none',
                    transition: 'filter 0.5s',
                  }} />
                <canvas ref={canvasRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
              </div>
            </div>

            <div style={{ display: 'flex', gap: 12, marginBottom: 24 }}>
              <button className="btn" onClick={analyze} disabled={loading}>
                <ScanLine size={16} />{loading ? 'Scanning...' : 'Run Diagnostic'}
              </button>
              <button className="btn" onClick={clear} disabled={loading}>Clear</button>
            </div>

            {error && <div className="alert error">{error}</div>}

            {result && (
              <div className="fade-in" style={{
                display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20,
                background: 'var(--bg-deep)', border: '1px solid var(--border)', padding: '28px 24px',
              }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
                  <RiskGauge score={result.score} threshold={threshold} size={220} showZones />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <div className={`verdict-block ${isFake ? 'fake' : 'real'}`} style={{ marginTop: 0 }}>
                    <div>
                      <div className="verdict-label">Forensic Verdict</div>
                      <div className="verdict-text">
                        {isFake
                          ? <><AlertTriangle size={18} style={{ display: 'inline', marginRight: 8 }} />Manipulation Detected</>
                          : <><CheckCircle size={18} style={{ display: 'inline', marginRight: 8 }} />Authentic</>}
                      </div>
                    </div>
                    <div className="verdict-score">
                      <div className="score-value">{Math.round(result.score * 100)}</div>
                      <div className="score-label">RISK SCORE</div>
                    </div>
                  </div>
                  <div style={{ marginTop: 16 }}>
                    {[
                      ['RAW SCORE',  `${(result.score * 100).toFixed(2)}%`],
                      ['CONFIDENCE', `${(result.confidence * 100).toFixed(1)}%`, true],
                      ['THRESHOLD',  `${(threshold * 100).toFixed(0)}%`],
                      ['MODEL',      'MobileNetV2'],
                      ['STATUS',     isFake ? '⚠ HIGH RISK' : '✓ CLEAR'],
                    ].map(([k, v, hi]) => (
                      <div className="stat-row" key={k}>
                        <span className="stat-key">{k}</span>
                        <span className="stat-val" style={hi ? { color: 'var(--cyan)' } : {}}>{v}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

