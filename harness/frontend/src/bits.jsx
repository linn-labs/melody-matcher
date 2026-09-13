// Small shared components for the experimental harness.

const { useState, useEffect, useRef, useMemo } = React;

// Publication curation: replace an unattributed brand glyph with a text label.
function AppleGlyph({ size = 16, color = "currentColor" }) {
  return <span aria-hidden="true" style={{ fontSize: size, color }}>XML</span>;
}

function IconPlay({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <path d="M4.5 3.2v9.6c0 .5.55.8.97.54l7.5-4.8a.64.64 0 0 0 0-1.08l-7.5-4.8A.64.64 0 0 0 4.5 3.2z"/>
    </svg>
  );
}

function IconChevron({ dir = "down", size = 12 }) {
  const rot = { down: 0, up: 180, left: 90, right: -90 }[dir] || 0;
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" style={{ transform: `rotate(${rot}deg)` }} aria-hidden="true">
      <path d="M2.5 4.25L6 7.75l3.5-3.5" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
    </svg>
  );
}

function IconThumbUp({ size = 14, filled = false }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"
         fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round">
      <path d="M4.5 7h-2v6h2V7zm0 0 3-5c.9 0 1.5.6 1.5 1.5V6h3.4c.9 0 1.5.8 1.35 1.67l-.78 4.3c-.13.72-.76 1.23-1.48 1.23H4.5V7z"/>
    </svg>
  );
}

function IconThumbDown({ size = 14, filled = false }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true"
         fill={filled ? "currentColor" : "none"} stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round">
      <path d="M4.5 9h-2V3h2v6zm0 0 3 5c.9 0 1.5-.6 1.5-1.5V10h3.4c.9 0 1.5-.8 1.35-1.67l-.78-4.3C11.84 3.31 11.21 2.8 10.5 2.8H4.5V9z"/>
    </svg>
  );
}

function IconFilter({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" aria-hidden="true">
      <path d="M2 4h12M4 8h8M6 12h4"/>
    </svg>
  );
}

function IconExternal({ size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M5 2H2.5v7.5H10V7M7 2h3v3M5 7l5-5"/>
    </svg>
  );
}

function IconCheck({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 8.5l3.5 3.5L13 5"/>
    </svg>
  );
}

function IconX({ size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" aria-hidden="true">
      <path d="M3 3l6 6M9 3l-6 6"/>
    </svg>
  );
}

function IconRefresh({ size = 12 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M10 2v3H7M2 10V7h3M10 5a4 4 0 0 0-7.4-1.5M2 7a4 4 0 0 0 7.4 1.5"/>
    </svg>
  );
}

function Artwork({ hue = 20, size = 44, title = "", radius = 10 }) {
  const c1 = `oklch(0.86 0.07 ${hue})`;
  const c2 = `oklch(0.72 0.10 ${(hue + 40) % 360})`;
  const letter = (title || "?").trim().charAt(0).toUpperCase();
  return (
    <div aria-hidden="true" style={{
      width: size, height: size, borderRadius: radius, flex: "0 0 auto",
      background: `linear-gradient(135deg, ${c1}, ${c2})`,
      boxShadow: "inset 0 0 0 1px rgba(0,0,0,0.04)",
      position: "relative", overflow: "hidden",
    }}>
      <div style={{
        position: "absolute", inset: 0,
        backgroundImage: "repeating-linear-gradient(135deg, rgba(255,255,255,0.06) 0 2px, transparent 2px 8px)",
      }} />
      <div style={{
        position: "absolute", inset: 0, display: "grid", placeItems: "center",
        color: "rgba(255,255,255,0.85)", fontFamily: '"Instrument Serif", serif',
        fontSize: size * 0.48, lineHeight: 1, letterSpacing: "-0.02em",
      }}>{letter}</div>
    </div>
  );
}

function ScoreBar({ value, width = 64 }) {
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        width, height: 4, borderRadius: 999,
        background: "oklch(0.93 0.01 70)", overflow: "hidden",
      }}>
        <div style={{
          width: `${pct * 100}%`, height: "100%",
          background: "var(--accent)",
          borderRadius: 999,
        }} />
      </div>
      <span className="mono" style={{ fontSize: 11, color: "var(--ink-soft)", minWidth: 36, textAlign: "right" }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

function Button({ children, onClick, variant = "default", size = "md", leading, trailing, disabled, title, style }) {
  const sizes = {
    sm: { padding: "6px 10px", fontSize: 12, borderRadius: 8, gap: 6 },
    md: { padding: "9px 14px", fontSize: 13, borderRadius: 10, gap: 8 },
    lg: { padding: "14px 22px", fontSize: 15, borderRadius: 14, gap: 10 },
  };
  const variants = {
    default: {
      background: "var(--card)", color: "var(--ink)",
      border: "1px solid var(--border-strong)",
      boxShadow: "var(--shadow-sm)",
    },
    ghost: { background: "transparent", color: "var(--ink-soft)", border: "1px solid transparent" },
    primary: {
      background: "var(--ink)", color: "var(--bg)",
      border: "1px solid var(--ink)",
      boxShadow: "var(--shadow-md)",
    },
    accent: {
      background: "var(--accent)", color: "white",
      border: "1px solid var(--accent)",
      boxShadow: "0 6px 18px oklch(0.58 0.15 28 / 0.30)",
    },
    soft: {
      background: "var(--accent-soft)", color: "var(--accent-ink)",
      border: "1px solid transparent",
    },
  };
  return (
    <button
      onClick={onClick} disabled={disabled} title={title}
      style={{
        display: "inline-flex", alignItems: "center", justifyContent: "center",
        cursor: disabled ? "default" : "pointer",
        fontWeight: 500, whiteSpace: "nowrap",
        transition: "transform 120ms ease, box-shadow 120ms ease, background 120ms ease",
        opacity: disabled ? 0.5 : 1,
        ...sizes[size], ...variants[variant], ...style,
      }}
      onMouseDown={(e) => { if (!disabled) e.currentTarget.style.transform = "translateY(1px)"; }}
      onMouseUp={(e) => { e.currentTarget.style.transform = ""; }}
      onMouseLeave={(e) => { e.currentTarget.style.transform = ""; }}
    >
      {leading}{children}{trailing}
    </button>
  );
}

function Chip({ children, active, onClick, trailing }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "inline-flex", alignItems: "center", gap: 6,
        padding: "5px 10px", borderRadius: 999, fontSize: 12,
        fontFamily: "inherit", cursor: "pointer",
        background: active ? "var(--accent-soft)" : "var(--card)",
        color: active ? "var(--accent-ink)" : "var(--ink-soft)",
        border: `1px solid ${active ? "transparent" : "var(--border)"}`,
        transition: "all 120ms ease",
      }}
    >{children}{trailing}</button>
  );
}

function Eyebrow({ children, style }) {
  return (
    <div style={{
      fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em",
      color: "var(--ink-mute)", fontWeight: 500, ...style,
    }}>{children}</div>
  );
}

function Spinner({ size = 14 }) {
  return (
    <span style={{
      width: size, height: size, borderRadius: "50%",
      border: "2px solid currentColor", borderRightColor: "transparent",
      display: "inline-block",
      animation: "harness-spin 700ms linear infinite",
    }} />
  );
}

function StatusBadge({ status }) {
  const map = {
    production: { label: "prod", color: "var(--good)", bg: "oklch(0.95 0.04 150)" },
    candidate:  { label: "cand", color: "oklch(0.48 0.10 240)", bg: "oklch(0.96 0.03 240)" },
    experimental: { label: "exp", color: "var(--accent-ink)", bg: "var(--accent-soft)" },
    baseline:   { label: "base", color: "var(--ink-soft)", bg: "oklch(0.94 0.006 70)" },
    unknown:    { label: "?",   color: "var(--ink-mute)", bg: "oklch(0.94 0.006 70)" },
  };
  const s = map[status] || map.baseline;
  return (
    <span className="mono" style={{
      fontSize: 9.5, textTransform: "uppercase", letterSpacing: "0.06em",
      padding: "2px 6px", borderRadius: 4,
      color: s.color, background: s.bg,
    }}>{s.label}</span>
  );
}

// Inject keyframes once
(function () {
  if (document.getElementById("harness-anim")) return;
  const s = document.createElement("style");
  s.id = "harness-anim";
  s.textContent = `
    @keyframes harness-spin { from { transform: rotate(0) } to { transform: rotate(360deg) } }
    @keyframes harness-fade { from { opacity: 0; transform: translateY(4px) } to { opacity: 1; transform: none } }
    @keyframes harness-pulse { 0%, 100% { opacity: 0.55 } 50% { opacity: 1 } }
  `;
  document.head.appendChild(s);
})();

Object.assign(window, {
  AppleGlyph, IconPlay, IconChevron, IconThumbUp, IconThumbDown,
  IconFilter, IconExternal, IconCheck, IconX, IconRefresh,
  Artwork, ScoreBar, Button, Chip, Eyebrow, Spinner, StatusBadge,
});
