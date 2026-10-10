import { useEffect, useRef, useState } from 'react';
import './DeepfakeGenerator.css';

interface InferenceResult {
  timestamp: number;
  session_id: string;
  frame_id: number;
  condition: string;
  face_detected: boolean;
  raw_model_a_similarity?: number;
  ema_model_a_similarity?: number;
  model_a_threshold?: number;
  model_a_raw_accept?: boolean;
  matched_identity?: string;
  raw_c_class?: number;
  raw_c_probs?: number[];
  smoothed_class?: number;
  ema_c_probs?: number[];
  raw_predicted_state?: string;
  final_state?: string;
  swap_latency?: number;
  model_a_latency?: number;
  model_c_latency?: number;
  total_latency?: number;
  fps?: number;
  status?: string;
  error?: string;
  processed_frame_b64?: string;
}

const CAPTURE_INTERVAL_MS = 1500; // conservative ~0.6 FPS max

export default function DeepfakeGenerator() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  
  const [streamActive, setStreamActive] = useState<boolean>(false);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');
  
  const [swapEnabled, setSwapEnabled] = useState<boolean>(false);
  const [enrollName, setEnrollName] = useState<string>('');
  const [enrollFiles, setEnrollFiles] = useState<FileList | null>(null);
  const [enrollStatus, setEnrollStatus] = useState<string>('');
  const [enrolledIdentity, setEnrolledIdentity] = useState<string>('Alice (Default)');

  // 1. Request Webcam Permission
  useEffect(() => {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false })
      .then(stream => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          setStreamActive(true);
        }
      })
      .catch(err => {
        console.error("Webcam access denied:", err);
        setErrorMsg("Webcam permission denied. Please allow access.");
      });
  }, []);

  // 2. Setup WebSocket (Dependent on Swap State)
  useEffect(() => {
    let isMounted = true;
    const condition = swapEnabled ? 'impersonation' : 'genuine';
    const wsUrl = `ws://localhost:8001/dl/ws/inference?condition=${condition}`;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout>;

    const connectWs = () => {
      if (!isMounted) return;
      ws = new WebSocket(wsUrl);
      
      ws.onopen = () => {
        if (!isMounted) {
           ws?.close();
           return;
        }
        setWsConnected(true);
        setErrorMsg('');
      };
      
      ws.onmessage = (event) => {
        if (!isMounted) return;
        try {
          const result: InferenceResult = JSON.parse(event.data);
          setInferenceResult(result);
        } catch (e) {
          console.error("Malformed JSON response", e);
        } finally {
          setIsProcessing(false);
        }
      };
      
      ws.onclose = () => {
        if (!isMounted) return;
        setWsConnected(false);
        setIsProcessing(false);
        wsRef.current = null;
        // Attempt reconnect after delay
        reconnectTimer = setTimeout(() => {
          connectWs();
        }, 3000);
      };
      
      ws.onerror = (e) => {
        console.error("WebSocket error:", e);
        ws?.close();
      };
      
      wsRef.current = ws;
    };
    
    // Add small delay to avoid strict mode immediately closing CONNECTING socket
    const startTimer = setTimeout(() => {
        connectWs();
    }, 100);
    
    return () => {
      isMounted = false;
      clearTimeout(startTimer);
      clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.onerror = null;
        if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN) {
            ws.close();
        }
      }
      wsRef.current = null;
      setIsProcessing(false);
    };
  }, [swapEnabled]);

  // 3. Periodic Capture Loop
  useEffect(() => {
    if (!streamActive || !wsConnected) return;
    
    const captureInterval = setInterval(() => {
      // Prevent accumulating outstanding frames
      if (isProcessing) return;
      
      if (videoRef.current && canvasRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        
        if (ctx && video.videoWidth > 0 && video.videoHeight > 0) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          canvas.toBlob((blob) => {
            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
              setIsProcessing(true);
              wsRef.current.send(blob);
            }
          }, 'image/jpeg', 0.85);
        }
      }
    }, CAPTURE_INTERVAL_MS);
    
    return () => clearInterval(captureInterval);
  }, [streamActive, wsConnected, isProcessing]);

  // UI Helpers
  const getStateColor = (state?: string) => {
    switch (state) {
      case 'VERIFIED': return 'state-verified';
      case 'SUSPECTED_IMPERSONATION': return 'state-suspected';
      default: return 'state-unknown';
    }
  };

  const handleEnroll = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!enrollName || !enrollFiles || enrollFiles.length === 0) {
      setEnrollStatus("Please provide a name and images.");
      return;
    }
    
    const formData = new FormData();
    formData.append("identity_name", enrollName);
    for (let i = 0; i < enrollFiles.length; i++) {
      formData.append("files", enrollFiles[i]);
    }
    
    setEnrollStatus("Enrolling...");
    try {
      const res = await fetch("http://localhost:8001/dl/enroll", {
        method: "POST",
        body: formData
      });
      const data = await res.json();
      if (res.ok) {
        setEnrollStatus(`✓ ${data.identity} enrolled successfully (References: ${data.references})`);
        setEnrolledIdentity(data.identity);
      } else {
        setEnrollStatus(`Error: ${data.detail}`);
      }
    } catch (err) {
      setEnrollStatus(`Error: ${err}`);
    }
  };

  return (
    <div className="dl-project-wrapper">
      <div className="container">
        <header className="generator-header">
        <h1>Adversarially Robust Face Identity Verification Against Synthetic Face-Swap Impersonation</h1>
        <div className="status-bar">
          <span className={`status-badge ${wsConnected ? 'connected' : 'disconnected'}`}>
            Backend: {wsConnected ? 'Connected' : 'Disconnected'}
          </span>
          <span className="status-badge">
            Status: {isProcessing ? 'Processing frame...' : 'Waiting for capture...'}
          </span>
        </div>
        {errorMsg && <div className="error-banner">{errorMsg}</div>}
      </header>
      
      <div className="main-content">
        <div className="controls-section">
          <div className="panel">
            <h2>Enroll New Identity</h2>
            <form onSubmit={handleEnroll} className="enroll-form">
              <div className="form-group">
                <label>Identity Name (SOURCE):</label>
                <input type="text" value={enrollName} onChange={e => setEnrollName(e.target.value)} placeholder="e.g. Rahul" />
              </div>
              <div className="form-group">
                <label>Reference Images (3-5 recommended):</label>
                <input type="file" multiple accept="image/*" onChange={e => setEnrollFiles(e.target.files)} />
              </div>
              <button type="submit" className="btn-primary">ENROLL IDENTITY</button>
            </form>
            {enrollStatus && <div className="enroll-status">{enrollStatus}</div>}
          </div>
          
          <div className="panel swap-panel">
            <h2>Face Swap Control</h2>
            <div className="swap-details">
              <p><strong>SOURCE (Enrolled Identity):</strong> {enrolledIdentity}</p>
              <p><strong>TARGET (Webcam Person):</strong> Live Webcam Person</p>
            </div>
            <div className="toggle-wrapper">
              <span>FACE SWAP</span>
              <button 
                className={`toggle-btn ${swapEnabled ? 'on' : 'off'}`}
                onClick={() => setSwapEnabled(!swapEnabled)}
              >
                {swapEnabled ? 'ON' : 'OFF'}
              </button>
            </div>
          </div>
          
          <div className="methodology-panel">
            <h3>Methodology</h3>
            <ul>
              <li><strong>Model A:</strong> Identity recognition gate (ArcFace)</li>
              <li><strong>C-Adv (legacy):</strong> placeholder output in this tab; the real detector is on the Unified Checkpoint page</li>
              <li><strong>Native InSwapper:</strong> Live face-swap test source (evaluated upstream)</li>
              <li>Final decision is produced purely by the server backend.</li>
              <li>When Face Swap is ON, the <strong>exact post-swap frame</strong> is displayed and analyzed.</li>
            </ul>
          </div>
        </div>
        
        <div className="video-section">
          <h2>{swapEnabled ? "POST-SWAP Live Preview" : "Live Webcam"}</h2>
          <div className="video-wrapper">
            {/* If swapped frame exists and swap is enabled, show it; otherwise show live webcam stream behind */}
            {swapEnabled && inferenceResult?.processed_frame_b64 ? (
              <img 
                src={`data:image/jpeg;base64,${inferenceResult.processed_frame_b64}`} 
                alt="Swapped Live Output" 
                className="swapped-image"
              />
            ) : null}
            <video 
              ref={videoRef} 
              autoPlay 
              playsInline 
              muted 
              className={swapEnabled && inferenceResult?.processed_frame_b64 ? 'hidden-video' : ''} 
            />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </div>
        </div>
        
        <div className="results-section">
          <h2>Inference Result</h2>
          
          <div className={`final-state-box ${getStateColor(inferenceResult?.final_state)}`}>
            <h3>FINAL STATE</h3>
            <div className="state-value">{inferenceResult?.final_state || 'WAITING'}</div>
            
            <div className="state-description">
              {inferenceResult?.final_state === 'VERIFIED' && 
                "identity accepted and anti-impersonation model accepts the sample as genuine"}
              {inferenceResult?.final_state === 'UNKNOWN' && 
                "identity gate or anti-impersonation decision did not establish a verified identity"}
              {inferenceResult?.final_state === 'SUSPECTED_IMPERSONATION' && 
                "anti-impersonation model explicitly flags impersonation"}
              {!inferenceResult?.final_state && "Waiting for first inference..."}
            </div>
          </div>
          
          {inferenceResult && (
            <div className="details-grid">
              <div className="detail-card">
                <h4>Face Detection</h4>
                <p>{inferenceResult.face_detected ? 'Detected' : 'No face found'}</p>
                {inferenceResult.error && <p className="error-text">{inferenceResult.error}</p>}
              </div>
              
              {inferenceResult.face_detected && (
                <>
                  <div className="detail-card">
                    <h4>Model A (Identity)</h4>
                    <p>Similarity: {inferenceResult.raw_model_a_similarity?.toFixed(4) || 'N/A'}</p>
                    <p>Threshold: {inferenceResult.model_a_threshold?.toFixed(4) || 'N/A'}</p>
                    <p>Accepted: {inferenceResult.model_a_raw_accept ? 'YES' : 'NO'}</p>
                  </div>
                  
                  <div className="detail-card">
                    <h4>C-Adv (Anti-Impersonation)</h4>
                    <p>Class: {inferenceResult.smoothed_class || inferenceResult.raw_c_class || 'N/A'}</p>
                    {inferenceResult.ema_c_probs && (
                      <div className="probs-list">
                        <small>G: {inferenceResult.ema_c_probs[0]?.toFixed(3)}</small>
                        <small>D: {inferenceResult.ema_c_probs[1]?.toFixed(3)}</small>
                        <small>I: {inferenceResult.ema_c_probs[2]?.toFixed(3)}</small>
                      </div>
                    )}
                    <p style={{ opacity: 0.7, fontSize: '0.75rem', marginTop: '0.4rem' }}>Legacy output: these values are fixed by the identity decision (G = 1 when ArcFace accepts), not a trained classifier. The real swap detector is on the Unified Checkpoint page.</p>
                  </div>
                  
                  <div className="detail-card">
                    <h4>Performance</h4>
                    <p>Total Latency: {inferenceResult.total_latency?.toFixed(2) || 'N/A'}s</p>
                    <p>FPS: {inferenceResult.fps?.toFixed(2) || 'N/A'}</p>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
        </div>
      </div>
    </div>
  );
}




