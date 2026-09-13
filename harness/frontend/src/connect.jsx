// Library import — drag/drop an Apple Music Library XML, POST to backend,
// show a match-rate confirmation, then hand the library handle up to App.

function ConnectHero({ onImport }) {
  const [phase, setPhase] = useState("idle");       // idle | uploading | matched | error
  const [report, setReport] = useState(null);       // backend response
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [showMisses, setShowMisses] = useState(false);
  const fileInput = useRef(null);

  const handleFile = async (file) => {
    if (!file) return;
    setPhase("uploading");
    setError(null);
    try {
      const r = await api.importLibrary(file, "aggressive");
      setReport({ ...r, filename: file.name });
      setPhase("matched");
    } catch (e) {
      setError(e.message || String(e));
      setPhase("error");
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const confirm = async () => {
    if (!report) return;
    try {
      const lib = await api.getLibrary(report.library_id);
      onImport({
        id: lib.id,
        user: lib.user || "you",
        display_name: lib.display_name,
        trackCount: lib.trackCount,
        matchedCount: lib.matchedCount,
        topCount: lib.topCount,
        source: "apple_music_xml",
        connectedAt: new Date(),
      });
    } catch (e) {
      setError(e.message || String(e));
      setPhase("error");
    }
  };

  return (
    <div style={{
      display: "grid", placeItems: "center", minHeight: "100vh",
      padding: "48px 24px", textAlign: "center",
    }}>
      <div style={{ maxWidth: 560, width: "100%" }}>
        <Eyebrow style={{ marginBottom: 18 }}>Recsys Harness · v0.1</Eyebrow>

        <h1 className="serif" style={{
          fontSize: 58, lineHeight: 1.04, margin: "0 0 14px",
          letterSpacing: "-0.02em", color: "var(--ink)",
        }}>
          Test recommendation<br/>models on <em>your</em> library.
        </h1>

        <p style={{
          fontSize: 15, lineHeight: 1.55, color: "var(--ink-soft)",
          margin: "0 auto 28px", maxWidth: 460,
        }}>
          Drop your Apple Music Library XML (File → Library → Export Library…
          in the Music desktop app). We fuzzy-match it against the 2.2M-track
          embedding catalog, then score candidates through a trained model.
        </p>

        {phase === "matched" && report
          ? <MatchReport report={report} onConfirm={confirm} onReset={() => { setPhase("idle"); setReport(null); }}
              showMisses={showMisses} onToggleMisses={() => setShowMisses(v => !v)} />
          : <DropZone
              dragOver={dragOver}
              phase={phase}
              error={error}
              onClick={() => fileInput.current?.click()}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            />
        }

        <input
          ref={fileInput}
          type="file" accept=".xml,.plist"
          style={{ display: "none" }}
          onChange={(e) => handleFile(e.target.files?.[0])}
        />

        {phase !== "matched" && (
          <div style={{
            marginTop: 32, display: "grid", gridTemplateColumns: "repeat(3, 1fr)",
            gap: 14, textAlign: "left",
          }}>
            <Feature n="01" title="Import library" body="Parse your Apple Music XML, extract tracks + playcounts." />
            <Feature n="02" title="Fuzzy match" body="Normalize, strip variant suffixes, match against 2.2M embeddings." />
            <Feature n="03" title="Inspect top 50" body="Score candidates through any trained model. Vote on each pick." />
          </div>
        )}
      </div>
    </div>
  );
}

function DropZone({ dragOver, phase, error, onClick, onDragOver, onDragLeave, onDrop }) {
  const uploading = phase === "uploading";
  return (
    <div
      onClick={onClick}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      style={{
        padding: "44px 24px",
        borderRadius: 18,
        border: `2px dashed ${dragOver ? "var(--accent)" : "var(--border-strong)"}`,
        background: dragOver ? "var(--accent-soft)" : "var(--card)",
        boxShadow: "var(--shadow-sm)",
        cursor: uploading ? "default" : "pointer",
        transition: "all 140ms ease",
      }}
    >
      <div style={{
        width: 52, height: 52, margin: "0 auto 14px",
        borderRadius: 14, display: "grid", placeItems: "center",
        background: "var(--bg-sunken)", border: "1px solid var(--border)",
        color: "var(--ink-soft)",
      }}>
        {uploading
          ? <Spinner size={20} />
          : <AppleGlyph size={20} />
        }
      </div>
      <div style={{ fontSize: 15, fontWeight: 600, marginBottom: 4 }}>
        {uploading ? "Parsing & matching…" : "Drop Library.xml here"}
      </div>
      <div style={{ fontSize: 12.5, color: "var(--ink-soft)" }}>
        {uploading
          ? "Fuzzy-matching against 2.2M track embeddings."
          : "Or click to choose a file. The file is sent to the harness backend."
        }
      </div>
      {error && (
        <div style={{
          marginTop: 16, padding: "10px 14px", borderRadius: 10,
          background: "oklch(0.96 0.04 25)", color: "var(--bad)",
          fontSize: 12, textAlign: "left",
        }}>{error}</div>
      )}
    </div>
  );
}

function MatchReport({ report, onConfirm, onReset, showMisses, onToggleMisses }) {
  const pct = report.total > 0 ? Math.round((report.matched / report.total) * 100) : 0;
  const lowMatch = pct < 50;
  return (
    <div style={{
      textAlign: "left",
      padding: 22, borderRadius: 18,
      background: "var(--card)", border: "1px solid var(--border)",
      boxShadow: "var(--shadow-md)",
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 14 }}>
        <div style={{
          width: 36, height: 36, borderRadius: 10,
          background: "var(--accent-soft)", color: "var(--accent-ink)",
          display: "grid", placeItems: "center",
        }}>
          <IconCheck size={18} />
        </div>
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ fontSize: 14, fontWeight: 600 }}>Library imported</div>
          <div className="mono" style={{
            fontSize: 11, color: "var(--ink-mute)",
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}>{report.filename}</div>
        </div>
      </div>

      <div style={{
        display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
        gap: 12, marginBottom: 14,
      }}>
        <Stat label="Imported" value={report.total.toLocaleString()} />
        <Stat label="Matched" value={report.matched.toLocaleString()} />
        <Stat label="Match rate" value={`${pct}%`} tone={lowMatch ? "bad" : "good"} />
      </div>

      {lowMatch && (
        <div style={{
          padding: "10px 12px", borderRadius: 10,
          background: "oklch(0.96 0.04 25)", color: "var(--bad)",
          fontSize: 12, marginBottom: 14,
        }}>
          Low match rate — inference will only see a fraction of your taste.
          Results may be under-served.
        </div>
      )}

      {report.sample_misses?.length > 0 && (
        <div style={{ marginBottom: 14 }}>
          <button onClick={onToggleMisses} style={{
            display: "inline-flex", alignItems: "center", gap: 6,
            background: "transparent", border: "none", cursor: "pointer",
            fontFamily: "inherit", color: "var(--ink-soft)", fontSize: 12,
            padding: 0,
          }}>
            <IconChevron dir={showMisses ? "up" : "down"} size={11} />
            {showMisses ? "Hide" : "Show"} unmatched sample ({report.sample_misses.length})
          </button>
          {showMisses && (
            <div style={{
              marginTop: 10, padding: 12, borderRadius: 10,
              background: "var(--bg-sunken)", border: "1px solid var(--border)",
              maxHeight: 200, overflowY: "auto",
            }}>
              {report.sample_misses.map((m, i) => (
                <div key={i} className="mono" style={{
                  fontSize: 11, color: "var(--ink-soft)",
                  padding: "3px 0",
                  whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                }}>
                  {m.artist} — {m.title}
                  {m.play_count ? ` · ${m.play_count} plays` : ""}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div style={{ display: "flex", gap: 10, justifyContent: "flex-end" }}>
        <Button variant="ghost" size="md" onClick={onReset}>Pick different file</Button>
        <Button variant="primary" size="md" onClick={onConfirm}
                leading={<IconCheck size={13} />}>
          Use this library
        </Button>
      </div>
    </div>
  );
}

function Stat({ label, value, tone }) {
  const color = tone === "bad" ? "var(--bad)" : tone === "good" ? "var(--good)" : "var(--ink)";
  return (
    <div style={{
      padding: 12, borderRadius: 10,
      background: "var(--bg-sunken)", border: "1px solid var(--border)",
    }}>
      <div style={{
        fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em",
        color: "var(--ink-mute)", fontWeight: 500, marginBottom: 4,
      }}>{label}</div>
      <div className="mono" style={{ fontSize: 18, color, fontWeight: 500 }}>
        {value}
      </div>
    </div>
  );
}

function Feature({ n, title, body }) {
  return (
    <div style={{
      padding: 18, borderRadius: 14, background: "var(--card)",
      border: "1px solid var(--border)", boxShadow: "var(--shadow-sm)",
    }}>
      <div className="mono" style={{ fontSize: 11, color: "var(--ink-mute)", marginBottom: 8 }}>{n}</div>
      <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 12.5, color: "var(--ink-soft)", lineHeight: 1.5 }}>{body}</div>
    </div>
  );
}

// Compact chip shown top-right once a library is loaded.
function LibraryChip({ library, onReset }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  useEffect(() => {
    const handler = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);
  const matchPct = library.trackCount > 0
    ? Math.round((library.matchedCount / library.trackCount) * 100)
    : 0;
  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        onClick={() => setOpen(v => !v)}
        style={{
          display: "inline-flex", alignItems: "center", gap: 10,
          padding: "7px 12px 7px 10px", borderRadius: 999,
          background: "var(--card)", border: "1px solid var(--border-strong)",
          boxShadow: "var(--shadow-sm)", cursor: "pointer",
          fontFamily: "inherit", color: "var(--ink)", fontSize: 12.5,
        }}
      >
        <span style={{
          width: 24, height: 24, borderRadius: "50%",
          background: "var(--ink)", color: "var(--bg)",
          display: "grid", placeItems: "center",
        }}>
          <AppleGlyph size={12} color="currentColor" />
        </span>
        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-start", lineHeight: 1.15 }}>
          <span style={{ fontSize: 10, color: "var(--ink-mute)", fontFamily: '"JetBrains Mono", monospace' }}>
            imported · lib {library.id}
          </span>
          <span style={{ fontSize: 12.5, fontWeight: 500 }}>
            {library.matchedCount.toLocaleString()} matched
          </span>
        </div>
        <IconChevron size={11} />
      </button>

      {open && (
        <div style={{
          position: "absolute", right: 0, top: "calc(100% + 8px)",
          background: "var(--card)", border: "1px solid var(--border)",
          borderRadius: 14, boxShadow: "var(--shadow-lg)", minWidth: 260,
          padding: 10, zIndex: 50,
        }}>
          <div style={{ padding: "8px 10px 12px", borderBottom: "1px solid var(--border)" }}>
            <div style={{ fontSize: 12, color: "var(--ink-mute)", marginBottom: 4 }}>Loaded</div>
            <div style={{ fontSize: 13, fontWeight: 600 }}>Apple Music (imported)</div>
            <div style={{ fontSize: 11.5, color: "var(--ink-soft)", marginTop: 2 }}>
              {library.trackCount.toLocaleString()} tracks · {library.matchedCount.toLocaleString()} matched ({matchPct}%)
            </div>
          </div>
          <button
            onClick={() => { onReset(); setOpen(false); }}
            style={menuItem}
          >
            <IconRefresh /> Import a different library
          </button>
        </div>
      )}
    </div>
  );
}

const menuItem = {
  display: "flex", alignItems: "center", gap: 10,
  width: "100%", padding: "9px 10px", borderRadius: 8,
  background: "transparent", border: "none", cursor: "pointer",
  fontFamily: "inherit", fontSize: 12.5, color: "var(--ink)",
  textAlign: "left",
};

Object.assign(window, { ConnectHero, LibraryChip });
