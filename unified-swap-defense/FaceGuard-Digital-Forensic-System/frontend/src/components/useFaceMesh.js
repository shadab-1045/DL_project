/**
 * useFaceMesh.js
 * Glassmorphism face detection HUD.
 * Inspired by: frosted glass card, rounded corners, soft cyan glow,
 * transparent glass panel floating over the face — premium, clean, Apple-like.
 */

import { useCallback } from 'react';

const easeOut = (t) => 1 - Math.pow(1 - t, 3);

export default function useFaceMesh() {
  const drawMeshOnCanvas = useCallback(async (source, canvas, opts = {}) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W   = canvas.width;
    const H   = canvas.height;
    const now = Date.now();

    const isFake   = opts.isFake   || false;
    const score    = opts.score    || 0;
    const hasResult = score > 0;

    // Color scheme
    const primary  = isFake ? '#ff2d6b' : '#00d4ff';
    const dimC     = isFake ? 'rgba(255,45,107,' : 'rgba(0,212,255,';
    const white    = 'rgba(255,255,255,';

    ctx.clearRect(0, 0, W, H);

    // ── 1. VERY SUBTLE VIGNETTE ───────────────────────────────────
    const breathe = 0.06 + 0.03 * Math.sin(now / 2200);
    const vig = ctx.createRadialGradient(W/2, H/2, H*0.2, W/2, H/2, H*0.8);
    vig.addColorStop(0,   'transparent');
    vig.addColorStop(0.7, dimC + (breathe * 0.3) + ')');
    vig.addColorStop(1,   dimC + (breathe * 0.8) + ')');
    ctx.fillStyle = vig;
    ctx.fillRect(0, 0, W, H);

    // Red tint when fake
    if (isFake) {
      ctx.fillStyle = 'rgba(255,20,60,0.14)';
      ctx.fillRect(0, 0, W, H);
    }

    // ── 2. SCAN LINE ─────────────────────────────────────────────
    const scanT = ((now % 5500) / 5500);
    const scanY = scanT * (H + 120) - 60;
    const sg    = ctx.createLinearGradient(0, scanY - 70, 0, scanY + 70);
    sg.addColorStop(0,   'transparent');
    sg.addColorStop(0.4, dimC + '0.04)');
    sg.addColorStop(0.5, dimC + (isFake ? '0.22' : '0.15') + ')');
    sg.addColorStop(0.6, dimC + '0.04)');
    sg.addColorStop(1,   'transparent');
    ctx.fillStyle = sg;
    ctx.fillRect(0, scanY - 70, W, 140);

    // ── 3. FACE DETECTION FRAME — glass card style ────────────────
    const isPortrait = H > W * 1.1;
    let fX, fY, fW, fH;
    if (isPortrait) {
      fW = W * 0.72; fH = H * 0.62;
      fX = (W - fW) / 2; fY = H * 0.05;
    } else {
      fW = W * 0.60; fH = H * 0.82;
      fX = (W - fW) / 2; fY = H * 0.05;
    }
    const fR = 14; // border radius

    const pulse = 0.75 + 0.25 * Math.sin(now / 900);

    // ── Outer glow ring (bloom effect) ──────────────────────────
    ctx.save();
    roundRect(ctx, fX - 3, fY - 3, fW + 6, fH + 6, fR + 3);
    ctx.shadowColor = primary;
    ctx.shadowBlur  = isFake ? 28 : 20;
    ctx.strokeStyle = isFake
      ? `rgba(255,45,107,${0.18 * pulse})`
      : `rgba(0,212,255,${0.15 * pulse})`;
    ctx.lineWidth = 8;
    ctx.stroke();
    ctx.restore();

    // ── Glass fill — frosted, transparent ──────────────────────
    ctx.save();
    roundRect(ctx, fX, fY, fW, fH, fR);
    ctx.clip();

    // Very transparent glass base
    ctx.fillStyle = isFake
      ? 'rgba(255,20,60,0.07)'
      : 'rgba(0,180,255,0.06)';
    ctx.fillRect(fX, fY, fW, fH);

    // Top glass shine (bright edge at top — key glassmorphism detail)
    const topShine = ctx.createLinearGradient(fX, fY, fX, fY + fH * 0.35);
    topShine.addColorStop(0,    white + '0.16)');
    topShine.addColorStop(0.25, white + '0.06)');
    topShine.addColorStop(0.6,  white + '0.01)');
    topShine.addColorStop(1,    'transparent');
    ctx.fillStyle = topShine;
    ctx.fillRect(fX, fY, fW, fH);

    // Diagonal shimmer (Apple glass light streak)
    const shimmer = ctx.createLinearGradient(fX, fY, fX + fW * 0.55, fY + fH * 0.4);
    shimmer.addColorStop(0,   white + '0.09)');
    shimmer.addColorStop(0.4, white + '0.03)');
    shimmer.addColorStop(1,   'transparent');
    ctx.fillStyle = shimmer;
    ctx.fillRect(fX, fY, fW, fH);

    ctx.restore();

    // ── Glass border — crisp, bright ───────────────────────────
    ctx.save();
    roundRect(ctx, fX, fY, fW, fH, fR);

    // Multi-layer border for depth
    // Layer 1: glow
    ctx.shadowColor = primary;
    ctx.shadowBlur  = isFake ? 18 : 12;
    ctx.strokeStyle = isFake
      ? `rgba(255,45,107,${0.55 * pulse})`
      : `rgba(0,212,255,${0.45 * pulse})`;
    ctx.lineWidth   = 1.5;
    ctx.stroke();

    // Layer 2: bright top edge (glass top highlight)
    ctx.shadowBlur  = 0;
    ctx.beginPath();
    ctx.moveTo(fX + fR, fY);
    ctx.lineTo(fX + fW - fR, fY);
    ctx.strokeStyle = white + (0.4 * pulse) + ')';
    ctx.lineWidth   = 1;
    ctx.stroke();

    ctx.restore();

    // ── 4. SUBTLE CORNER INDICATORS ─────────────────────────────
    const cSize = 14;
    ctx.shadowColor = primary;
    ctx.shadowBlur  = 10;
    ctx.strokeStyle = isFake
      ? `rgba(255,45,107,${0.85 * pulse})`
      : `rgba(0,212,255,${0.8 * pulse})`;
    ctx.lineWidth   = 2;
    ctx.lineCap     = 'round';

    // Just the very tips of corners — minimal, elegant
    [[fX, fY, 1, 1], [fX+fW, fY, -1, 1], [fX, fY+fH, 1, -1], [fX+fW, fY+fH, -1, -1]].forEach(([x, y, dx, dy]) => {
      ctx.beginPath();
      ctx.moveTo(x + dx * cSize, y);
      ctx.lineTo(x + dx * 2, y + dy * 2);
      ctx.lineTo(x, y + dy * cSize);
      ctx.stroke();
    });

    // ── 5. CENTER CROSSHAIR ──────────────────────────────────────
    const ccx = fX + fW / 2;
    const ccy = fY + fH * (isPortrait ? 0.37 : 0.41);
    const chL = fW * 0.032;

    ctx.shadowBlur  = 6;
    ctx.strokeStyle = white + (0.28 * pulse) + ')';
    ctx.lineWidth   = 0.8;
    ctx.beginPath(); ctx.moveTo(ccx - chL, ccy); ctx.lineTo(ccx + chL, ccy); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ccx, ccy - chL); ctx.lineTo(ccx, ccy + chL); ctx.stroke();
    ctx.beginPath();
    ctx.arc(ccx, ccy, 2.5, 0, Math.PI * 2);
    ctx.strokeStyle = dimC + (0.5 * pulse) + ')';
    ctx.lineWidth   = 1.2;
    ctx.stroke();

    // ── 6. EXPANDING RINGS ───────────────────────────────────────
    const ringT  = (now % 3000) / 3000;
    const maxR   = Math.min(fW, fH) * 0.28;
    for (let i = 0; i < 2; i++) {
      const t  = (easeOut(ringT) + i * 0.5) % 1;
      const rr = t * maxR;
      const ra = (1 - t) * (isFake ? 0.38 : 0.22);
      if (ra < 0.01) continue;
      ctx.beginPath();
      ctx.arc(ccx, ccy, rr, 0, Math.PI * 2);
      ctx.strokeStyle = dimC + ra + ')';
      ctx.lineWidth   = 1;
      ctx.shadowColor = primary; ctx.shadowBlur = 8;
      ctx.stroke();
    }

    // ── 7. GLASS DATA PANEL — inside frame, bottom ───────────────
    const pH  = fH * 0.22;
    const pW  = fW * 0.82;
    const pX  = fX + (fW - pW) / 2;
    const pY  = fY + fH - pH - fH * 0.05;
    const pR  = 10;

    // ── GlassSurface-style panel ─────────────────────────────────
    // Mimics: rgba(255,255,255,0.1) base + inset white borders + color tint
    ctx.save();
    roundRect(ctx, pX, pY, pW, pH, pR);
    ctx.clip();

    // White glass base (GlassSurface fallback dark mode: rgba(255,255,255,0.1))
    ctx.fillStyle = 'rgba(255,255,255,0.08)';
    ctx.fillRect(pX, pY, pW, pH);

    // Subtle color tint over the white glass
    ctx.fillStyle = isFake
      ? 'rgba(255,45,107,0.07)'
      : 'rgba(0,212,255,0.06)';
    ctx.fillRect(pX, pY, pW, pH);

    // Top inset highlight (GlassSurface: inset 0 1px 0 rgba(255,255,255,0.2))
    const pShine = ctx.createLinearGradient(pX, pY, pX, pY + pH * 0.45);
    pShine.addColorStop(0,    white + '0.22)');
    pShine.addColorStop(0.25, white + '0.09)');
    pShine.addColorStop(0.7,  white + '0.02)');
    pShine.addColorStop(1,    'transparent');
    ctx.fillStyle = pShine;
    ctx.fillRect(pX, pY, pW, pH);

    // Bottom inset shadow (GlassSurface: inset 0 -1px 0 rgba(255,255,255,0.1))
    const pBottom = ctx.createLinearGradient(pX, pY + pH * 0.7, pX, pY + pH);
    pBottom.addColorStop(0,   'transparent');
    pBottom.addColorStop(1,   white + '0.07)');
    ctx.fillStyle = pBottom;
    ctx.fillRect(pX, pY, pW, pH);

    ctx.restore();

    // Panel border — white glass border (GlassSurface: border 1px solid rgba(255,255,255,0.2))
    ctx.save();
    roundRect(ctx, pX, pY, pW, pH, pR);
    ctx.strokeStyle = isFake
      ? 'rgba(255,45,107,0.35)'
      : 'rgba(255,255,255,0.18)';
    ctx.lineWidth   = 1;
    ctx.shadowColor = primary; ctx.shadowBlur = 8;
    ctx.stroke();
    // Bright top edge
    ctx.beginPath();
    ctx.moveTo(pX + pR, pY); ctx.lineTo(pX + pW - pR, pY);
    ctx.strokeStyle = white + (0.45 * pulse) + ')';
    ctx.shadowBlur  = 0; ctx.lineWidth = 1;
    ctx.stroke();
    ctx.restore();

    // Left accent line
    ctx.fillStyle   = isFake ? 'rgba(255,45,107,0.7)' : dimC + '0.7)';
    ctx.shadowColor = primary; ctx.shadowBlur = 6;
    ctx.fillRect(pX, pY + pH * 0.22, 2, pH * 0.56);

    // Panel text
    ctx.shadowBlur = 0;
    const tx     = pX + 14;
    const baseFZ = Math.max(9, Math.round(pH * 0.18));
    const bigFZ  = Math.max(11, Math.round(pH * 0.26));

    ctx.font      = `bold ${baseFZ}px 'Share Tech Mono', monospace`;
    ctx.fillStyle = white + '0.38)';
    ctx.fillText('BIOMETRIC  SCAN', tx, pY + pH * 0.3);

    ctx.font = `bold ${bigFZ}px 'Share Tech Mono', monospace`;
    if (!hasResult) {
      const p2 = 0.55 + 0.45 * Math.sin(now / 600);
      ctx.fillStyle = white + (0.72 * p2) + ')';
      ctx.fillText('⟳  ANALYZING...', tx, pY + pH * 0.6);
    } else {
      ctx.fillStyle = isFake ? 'rgba(255,80,110,0.97)' : 'rgba(0,220,255,0.96)';
      ctx.fillText(
        isFake ? `⚠  RISK: ${Math.round(score * 100)}%` : `✓  AUTHENTIC  ${Math.round((1-score)*100)}%`,
        tx, pY + pH * 0.6
      );
    }

    ctx.font      = `${Math.max(8, Math.round(pH * 0.14))}px 'Share Tech Mono', monospace`;
    ctx.fillStyle = white + '0.24)';
    ctx.fillText('FACEGUARD v3.0  ·  MobileNetV2', tx, pY + pH * 0.86);

    // Status dot
    const dotP = 0.5 + 0.5 * Math.sin(now / 450);
    const dotX = pX + pW - 16; const dotY = pY + pH * 0.37;
    const dotC = !hasResult
      ? dimC + (0.45 + 0.3 * dotP) + ')'
      : isFake ? `rgba(255,45,107,${dotP})` : `rgba(0,230,118,${dotP})`;
    ctx.beginPath(); ctx.arc(dotX, dotY, 4.5, 0, Math.PI * 2);
    ctx.fillStyle = dotC; ctx.shadowColor = dotC; ctx.shadowBlur = 12; ctx.fill();
    ctx.beginPath(); ctx.arc(dotX, dotY, 8, 0, Math.PI * 2);
    ctx.strokeStyle = dotC; ctx.lineWidth = 0.6; ctx.shadowBlur = 0; ctx.stroke();

    // ── 8. LABEL ─────────────────────────────────────────────────
    ctx.shadowBlur  = 0;
    ctx.font        = `bold ${Math.max(8, Math.round(H * 0.016))}px 'Share Tech Mono', monospace`;
    ctx.fillStyle   = white + '0.42)';
    ctx.fillText('FACE  DETECTION', fX + 6, Math.max(fY - 6, 12));
    ctx.textAlign   = 'right';
    ctx.fillStyle   = dimC + '0.42)';
    ctx.fillText('v3.0', fX + fW - 6, Math.max(fY - 6, 12));
    ctx.textAlign   = 'left';

    // ── 9. EDGE BLOOM ────────────────────────────────────────────
    const bb = ctx.createLinearGradient(0, H - 2, W, H - 2);
    bb.addColorStop(0,    'transparent');
    bb.addColorStop(0.25, dimC + (breathe * 3.5) + ')');
    bb.addColorStop(0.5,  dimC + (breathe * 5.0) + ')');
    bb.addColorStop(0.75, dimC + (breathe * 3.5) + ')');
    bb.addColorStop(1,    'transparent');
    ctx.fillStyle = bb; ctx.fillRect(0, H - 4, W, 4);

  }, []);

  return { drawMeshOnCanvas, ready: true };
}

// Helper: rounded rect path
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
