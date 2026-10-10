import React, { useState, useEffect } from 'react';
import './index.css';
import LandingPage from './components/LandingPage';
import ImageAnalysis from './components/ImageAnalysis';
import VideoForensics from './components/VideoForensics';
import WebcamScan from './components/WebcamScan';
import DeepfakeGenerator from './components/DeepfakeGenerator';
import UnifiedCheckpoint from './components/UnifiedCheckpoint';
import { Image, Moon, Sun, Film, Camera, ArrowLeft, RefreshCw, ShieldCheck } from 'lucide-react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8001/faceguard';

const MODES = [
  { id: 'image',  label: 'Image Analysis',   Icon: Image  },
  { id: 'video',  label: 'Video Forensics',   Icon: Film   },
  { id: 'webcam', label: 'Live Surveillance', Icon: Camera },
  { id: 'generator', label: 'Deepfake Generator', Icon: RefreshCw },
  { id: 'unified', label: 'Unified Checkpoint', Icon: ShieldCheck },
];

export default function App() {
  const [view, setView]             = useState('landing');
  const [theme, setTheme]           = useState('linear');
  const [mode, setMode]             = useState('image');
  const [threshold, setThreshold]   = useState(0.40);
  const [apiOnline, setApiOnline]   = useState(null);

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  useEffect(() => {
    fetch(`${API}/health`)
      .then(r => r.json())
      .then(d => setApiOnline(d.model_loaded === true))
      .catch(() => setApiOnline(false));
  }, []);

  const enterApp = (selectedMode) => { setMode(selectedMode); setView('app'); };
  const backToLanding = () => { setView('landing'); };
  const switchMode = (newMode) => { setMode(newMode); };

  return (
    <>
      {view === 'landing' && (
        <div className="view-wrap">
          <LandingPage onEnter={enterApp} theme={theme} setTheme={setTheme} apiOnline={apiOnline} />
        </div>
      )}

      {view === 'app' && (
        <div className="view-wrap app-shell">
          <div className="app-grid" />

          <header className="header">
            <div className="logo">
              <button className="back-btn" onClick={backToLanding} title="Back to home">
                <ArrowLeft size={15} />
              </button>
              <div className="logo-icon">
                <svg width="18" height="18" viewBox="0 0 18 18" fill="none">
                  <path d="M9 1L16.5 5v8L9 17 1.5 13V5L9 1z" stroke="currentColor" strokeWidth="1.2" fill="none" />
                  <path d="M9 5l4.5 2.5v5L9 15 4.5 12.5v-5L9 5z" fill="currentColor" opacity="0.2" stroke="currentColor" strokeWidth="0.8" />
                </svg>
              </div>
              <div>
                <div className="logo-text"><span>FACE</span>GUARD</div>
                <div className="logo-version">FORENSIC ENGINE v3.0 — NEURAL DEEPFAKE DETECTION</div>
              </div>
            </div>
            
            <div style={{display: 'flex', alignItems: 'center'}}>
              <button className="theme-toggle-btn" onClick={() => setTheme(theme === 'linear' ? 'vercel' : 'linear')} style={{ background: 'transparent', border: '1px solid var(--border)', borderRadius: '4px', padding: '4px 8px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '6px', marginRight: '16px' }}>
                {theme === 'linear' ? <Sun size={14} /> : <Moon size={14} />}
                <span style={{ fontSize: '0.7rem' }}>{theme === 'linear' ? 'Light' : 'Dark'}</span>
              </button>
              <div className="header-status">
                <div className="status-dot" style={{
                  background: apiOnline === null ? 'var(--theme-amber)' : apiOnline ? 'var(--theme-success)' : 'var(--theme-danger)',
                  boxShadow: `0 0 8px ${apiOnline === null ? 'var(--theme-amber)' : apiOnline ? 'var(--theme-success)' : 'var(--theme-danger)'}`,
                }} />
                {apiOnline === null ? 'CONNECTING...' : apiOnline ? 'NEURAL ENGINE ONLINE' : 'API OFFLINE'}
              </div>
            </div>
          </header>

          <main className="main">
            <aside className="sidebar">
              <div className="sidebar-label">Mode Select</div>
              {MODES.map(({ id, label, Icon }) => (
                <button
                  key={id}
                  className={`nav-btn ${mode === id ? 'active' : ''}`}
                  onClick={() => switchMode(id)}
                >
                  <Icon className="icon" size={15} />
                  {label}
                </button>
              ))}

              <div className="sidebar-label">Settings</div>
              <div className="threshold-control">
                <div className="threshold-label">
                  Confidence Threshold
                  <span>{(threshold * 100).toFixed(0)}%</span>
                </div>
                <input
                  className="threshold-slider"
                  type="range" min={0.1} max={0.9} step={0.05}
                  value={threshold}
                  onChange={e => setThreshold(parseFloat(e.target.value))}
                />
              </div>

              {apiOnline === false && (
                <div className="alert error" style={{ marginTop: 16, fontSize: '0.6rem' }}>
                  BACKEND OFFLINE
                </div>
              )}

              </aside>

            <section className="content">
              <div className="mode-wrap">
                {mode === 'image'  && <ImageAnalysis threshold={threshold} />}
                {mode === 'video'  && <VideoForensics threshold={threshold} />}
                {mode === 'webcam' && <WebcamScan threshold={threshold} />}
                {mode === 'generator' && <DeepfakeGenerator />}
                {mode === 'unified' && <UnifiedCheckpoint />}
              </div>
            </section>
          </main>

          <nav className="mobile-nav">
            {MODES.map(({ id, label, Icon }) => (
              <button
                key={id}
                className={`mobile-nav-btn ${mode === id ? 'active' : ''}`}
                onClick={() => switchMode(id)}
              >
                <Icon size={20} />
                <span>{label.split(' ')[0]}</span>
              </button>
            ))}
          </nav>
        </div>
      )}
    </>
  );
}




