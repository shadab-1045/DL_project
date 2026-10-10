import React, { useState } from 'react';
import './LandingPage.css';
import { Image, Film, Camera, Info, X, Shield, Moon, Sun, ShieldCheck } from 'lucide-react';

const MODES = [
  {
    id: 'unified',
    Icon: ShieldCheck,
    label: 'Unified Checkpoint',
    tagline: 'ATTACK VS DEFENSE',
    desc: 'ArcFace identity check plus a face-swap detector. Toggle a live InSwapper attack and watch the detector veto the match.',
    color: '#ff2d6b',
    rgb: '255,45,107',
  },
  {
    id: 'image',
    Icon: Image,
    label: 'Image Analysis',
    tagline: 'FORENSIC SCAN',
    desc: 'Score a single image with the FaceGuard classifier (trained on FaceForensics++; see README for measured accuracy).',
    color: '#00d4ff',
    rgb: '0,212,255',
  },
  {
    id: 'video',
    Icon: Film,
    label: 'Video Forensics',
    tagline: 'TEMPORAL ANALYSIS',
    desc: 'Upload a video and see the per-frame FaceGuard fake score as a timeline.',
    color: '#a855f7',
    rgb: '168,85,247',
  },
  {
    id: 'webcam',
    Icon: Camera,
    label: 'Live Surveillance',
    tagline: 'REAL-TIME SCAN',
    desc: 'Stream webcam frames to the FaceGuard classifier over WebSocket and watch the score update live.',
    color: '#00e676',
    rgb: '0,230,118',
  },
];




export default function LandingPage({ onEnter, theme, setTheme, apiOnline }) {
  const [hovered, setHovered] = useState(null);
  

  return (
    <div className="landing">
      
      <div className="landing-grain" />
      <div className="landing-grid" />

      {/* Header */}
      <header className="landing-header">
        <div className="landing-logo">
          <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
            <path d="M16 2L29 9v14L16 30 3 23V9L16 2z" stroke="#00d4ff" strokeWidth="1.5" fill="rgba(0,212,255,0.06)" />
            <path d="M16 8L24 12.5v9L16 26 8 21.5v-9L16 8z" fill="rgba(0,212,255,0.12)" stroke="#00d4ff" strokeWidth="1" />
            <circle cx="16" cy="16" r="3" fill="#00d4ff" opacity="0.8" />
          </svg>
          <div>
            <div className="landing-logo-text"><span>FACE</span>GUARD</div>
            <div className="landing-logo-sub">FORENSIC ENGINE v3.0</div>
          </div>
        </div>
        
              <button className="landing-about-btn" onClick={() => setTheme(theme === 'linear' ? 'vercel' : 'linear')} style={{ padding: '6px 12px', background: 'transparent', border: '1px solid var(--border)', borderRadius: '6px', color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: '8px' }}>
          {theme === 'linear' ? <Sun size={15} /> : <Moon size={15} />}
          <span>{theme === 'linear' ? 'Light' : 'Dark'}</span>
        </button>
      </header>

      {/* Main */}
      <main className="landing-main">
        <div className="landing-hero">
          <div className="landing-badge">
            <span className="landing-badge-dot" />
            {apiOnline === null ? 'CONNECTING...' : apiOnline ? 'BACKEND ONLINE' : 'BACKEND OFFLINE - START uvicorn ON :8001'}
          </div>
          <h1 className="landing-title">
            Attack it.<br />
            <span className="landing-title-accent">Then defend it.</span>
          </h1>
          <p className="landing-subtitle">
            Face-swap impersonation vs. identity verification, with a measured, adversarially-tested defense.<br />
            Select a mode to begin.
          </p>
        </div>

        {/* Mode Cards */}
        <div className="landing-cards">
          {MODES.map(({ id, Icon, label, tagline, desc, color, rgb }, i) => (
            <button
              key={id}
              className={`landing-card ${hovered === id ? 'landing-card--hovered' : ''}`}
              style={{ '--card-color': color, '--card-rgb': rgb, animationDelay: `${i * 0.12}s` }}
              onMouseEnter={() => setHovered(id)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => onEnter(id)}
            >
              {/* Glass layers */}
              <div className="lcard-glass-base" />
              <div className="lcard-glass-shine" />
              <div className="lcard-glass-border" />
              <div className="lcard-glow" />

              <div className="lcard-inner">
                <div className="lcard-top">
                  <span className="lcard-tag">{tagline}</span>
                  <div className="lcard-dot" />
                </div>
                <div className="lcard-icon"><Icon size={30} /></div>
                <h2 className="lcard-title">{label}</h2>
                <p className="lcard-desc">{desc}</p>
                <div className="lcard-cta">
                  <span>ACTIVATE</span>
                  <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                    <path d="M3 8h10M9 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              </div>

              {/* Corner brackets */}
              <div className="lcard-corner lcard-corner--tl" />
              <div className="lcard-corner lcard-corner--tr" />
              <div className="lcard-corner lcard-corner--bl" />
              <div className="lcard-corner lcard-corner--br" />
            </button>
          ))}
        </div>

        
      </main>

      {/* Footer */}
    </div>
  );
}











