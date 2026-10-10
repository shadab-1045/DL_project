import React from 'react';

export default function RiskGauge({ score = 0, threshold = 0.40, size = 200, showZones = false }) {
  const cx = size / 2;
  const cy = size / 2;
  const r = (size / 2) * 0.76;
  const sw = size * 0.056;

  const startAngle = -210;
  const totalDeg = 240;
  const toRad = (d) => (d * Math.PI) / 180;

  const polar = (angle, radius) => ({
    x: cx + radius * Math.cos(toRad(angle)),
    y: cy + radius * Math.sin(toRad(angle)),
  });

  const arc = (start, end, rad) => {
    const s = polar(start, rad); const e = polar(end, rad);
    const large = end - start > 180 ? 1 : 0;
    return `M ${s.x} ${s.y} A ${rad} ${rad} 0 ${large} 1 ${e.x} ${e.y}`;
  };

  const filledAngle = startAngle + totalDeg * Math.min(score, 1);
  const thresholdAngle = startAngle + totalDeg * threshold;
  const isFake = score > threshold;

  // Color: interpolate cyan → amber → red based on score
  const hue = isFake ? Math.max(340 - (score - threshold) / (1 - threshold) * 60, 280) : 195;
  const fillColor = isFake
    ? `hsl(${hue}, 90%, 60%)`
    : `hsl(${195 - score * 60}, 85%, 55%)`;

  // Needle
  const needleAngle = startAngle + totalDeg * Math.min(score, 1);
  const nTip = polar(needleAngle, r - sw / 2 - 4);
  const nB1 = polar(needleAngle + 90, 5);
  const nB2 = polar(needleAngle - 90, 5);

  // Threshold tick
  const tOuter = polar(thresholdAngle, r + sw / 2 + 5);
  const tInner = polar(thresholdAngle, r - sw / 2 - 5);

  // Zone arc — danger zone from threshold to max
  const dangerStart = startAngle + totalDeg * threshold;
  const dangerEnd = startAngle + totalDeg * 1.0;

  // Tick marks
  const ticks = Array.from({ length: 13 }, (_, i) => {
    const a = startAngle + totalDeg * (i / 12);
    const outer = polar(a, r + sw / 2 + 3);
    const inner = polar(a, r + sw / 2 + (i % 3 === 0 ? 10 : 6));
    return { outer, inner, major: i % 3 === 0, a, val: Math.round(i / 12 * 100) };
  });

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10 }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} overflow="visible">
        <defs>
          <radialGradient id="gaugeGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor={fillColor} stopOpacity="0.08" />
            <stop offset="100%" stopColor={fillColor} stopOpacity="0" />
          </radialGradient>
          <filter id="glow">
            <feGaussianBlur stdDeviation="2.5" result="coloredBlur" />
            <feMerge><feMergeNode in="coloredBlur" /><feMergeNode in="SourceGraphic" /></feMerge>
          </filter>
        </defs>

        {/* Background glow */}
        <circle cx={cx} cy={cy} r={r + sw} fill="url(#gaugeGlow)" />

        {/* Track */}
        <path d={arc(startAngle, startAngle + totalDeg, r)} fill="none"
          stroke="rgba(0,229,255,0.07)" strokeWidth={sw} strokeLinecap="round" />

        {/* Danger zone tint */}
        {showZones && (
          <path d={arc(dangerStart, dangerEnd, r)} fill="none"
            stroke="rgba(255,45,107,0.1)" strokeWidth={sw} strokeLinecap="round" />
        )}

        {/* Filled arc */}
        {score > 0.008 && (
          <path d={arc(startAngle, filledAngle, r)} fill="none"
            stroke={fillColor} strokeWidth={sw} strokeLinecap="round"
            filter="url(#glow)"
            style={{ transition: 'all 0.3s ease' }}
          />
        )}

        {/* Tick marks */}
        {ticks.map(({ outer, inner, major, val, a }, i) => (
          <g key={i}>
            <line x1={inner.x} y1={inner.y} x2={outer.x} y2={outer.y}
              stroke={a <= filledAngle ? fillColor : 'rgba(0,229,255,0.2)'}
              strokeWidth={major ? 1.5 : 0.8}
            />
          </g>
        ))}

        {/* Threshold marker */}
        <line x1={tInner.x} y1={tInner.y} x2={tOuter.x} y2={tOuter.y}
          stroke="#ffab00" strokeWidth={2.5}
          style={{ filter: 'drop-shadow(0 0 4px #ffab00)' }}
        />

        {/* Danger zone label */}
        {showZones && (() => {
          const labelAngle = startAngle + totalDeg * (threshold + (1 - threshold) * 0.5);
          const lp = polar(labelAngle, r + sw * 1.9);
          return (
            <text x={lp.x} y={lp.y} textAnchor="middle"
              fill="rgba(255,45,107,0.45)" fontFamily="'Share Tech Mono', monospace"
              fontSize={size * 0.045} transform={`rotate(${labelAngle + 90}, ${lp.x}, ${lp.y})`}>
              DANGER
            </text>
          );
        })()}

        {/* Threshold label */}
        {showZones && (() => {
          const lp = polar(thresholdAngle, r - sw * 2.2);
          return (
            <text x={lp.x} y={lp.y} textAnchor="middle"
              fill="rgba(255,171,0,0.7)" fontFamily="'Share Tech Mono', monospace"
              fontSize={size * 0.042}>
              {Math.round(threshold * 100)}%
            </text>
          );
        })()}

        {/* Needle */}
        <polygon
          points={`${nTip.x},${nTip.y} ${nB1.x},${nB1.y} ${nB2.x},${nB2.y}`}
          fill={fillColor}
          style={{ filter: `drop-shadow(0 0 5px ${fillColor})`, transition: 'all 0.3s ease' }}
        />

        {/* Center cap — glassy */}
        <circle cx={cx} cy={cy} r={size * 0.06} fill="var(--bg-panel)"
          stroke={fillColor} strokeWidth={1.5}
          style={{ filter: `drop-shadow(0 0 4px ${fillColor})` }} />
        <circle cx={cx} cy={cy} r={size * 0.03} fill={fillColor} opacity={0.6} />

        {/* Score number */}
        <text x={cx} y={cy + r * 0.46} textAnchor="middle"
          fill={fillColor} fontFamily="'Share Tech Mono', monospace"
          fontSize={size * 0.16} style={{ transition: 'fill 0.3s ease' }}>
          {Math.round(score * 100)}
        </text>

        <text x={cx} y={cy + r * 0.62} textAnchor="middle"
          fill="var(--text-muted)" fontFamily="'Share Tech Mono', monospace"
          fontSize={size * 0.052} letterSpacing="2">
          RISK %
        </text>

        {/* Min / max labels */}
        {[
          [startAngle, '0'],
          [startAngle + totalDeg, '100'],
        ].map(([a, label], i) => {
          const p = polar(a, r + sw + 12);
          return (
            <text key={i} x={p.x} y={p.y + 4} textAnchor="middle"
              fill="var(--text-muted)" fontFamily="'Share Tech Mono', monospace"
              fontSize={size * 0.048}>
              {label}
            </text>
          );
        })}
      </svg>

      {/* Status label below gauge */}
      <div style={{
        fontFamily: "'Share Tech Mono', monospace",
        fontSize: size * 0.052 + 'px',
        letterSpacing: 2,
        color: isFake ? 'var(--magenta)' : 'var(--green)',
        textTransform: 'uppercase',
        textAlign: 'center',
        filter: `drop-shadow(0 0 6px ${isFake ? 'var(--magenta)' : 'var(--green)'})`,
      }}>
        {isFake ? '⚠ MANIPULATION DETECTED' : '✓ SIGNAL AUTHENTIC'}
      </div>
    </div>
  );
}
