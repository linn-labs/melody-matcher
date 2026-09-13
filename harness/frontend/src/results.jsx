// Results panel: header, filters, 50-row list with score + feedback + play link.
// Rows from the backend contain: rank, score, embedding_index, embedding_key,
// artist, title — and sometimes a "vote" string ("up"|"down") when rehydrated.
// Optional fields (album, year, genre, explicit, duration) are gracefully
// hidden when absent.

function ResultsPanel({ run, model, cardStyle, onCardStyleChange, onRerun, running, onVote, compare }) {
  const [filters, setFilters] = useState({
    genre: "all",
    year: "all",        // all | since2015 | since2020
    query: "",
  });

  // Reset scroll when flipping carousel slots — different track sets, so the
  // previous scroll position is meaningless.
  useEffect(() => {
    if (!compare) return;
    window.scrollTo({ top: 0, behavior: "instant" });
  }, [compare && compare.idx]);

  const availableGenres = useMemo(() => {
    if (!run) return [];
    const s = new Set();
    for (const r of run.results) if (r.genre) s.add(r.genre);
    return Array.from(s).sort();
  }, [run]);

  const anyYear    = useMemo(() => !!run && run.results.some(r => r.year),    [run]);
  const anyGenre   = availableGenres.length > 0;

  const filtered = useMemo(() => {
    if (!run) return [];
    return run.results.filter(r => {
      if (filters.genre !== "all" && r.genre !== filters.genre) return false;
      if (filters.year === "since2020" && !(r.year >= 2020)) return false;
      if (filters.year === "since2015" && !(r.year >= 2015)) return false;
      if (filters.query.trim()) {
        const q = filters.query.toLowerCase();
        if (!(r.title || "").toLowerCase().includes(q) &&
            !(r.artist || "").toLowerCase().includes(q)) return false;
      }
      return true;
    });
  }, [run, filters]);

  if (!run) {
    if (compare) {
      return (
        <main style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>
          <CompareCarousel compare={compare} />
          <div style={{ flex: 1, display: "grid", placeItems: "center", padding: 48 }}>
            <CompareSlotLoading compare={compare} />
          </div>
        </main>
      );
    }
    return (
      <main style={{ flex: 1, display: "grid", placeItems: "center", padding: 48 }}>
        <EmptyState onRerun={onRerun} running={running} />
      </main>
    );
  }

  const up   = run.feedback?.up   ?? 0;
  const down = run.feedback?.down ?? 0;

  return (
    <main style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", animation: "harness-fade 260ms ease" }}>
      {compare && <CompareCarousel compare={compare} />}
      <div style={{
        padding: "26px 40px 18px",
        borderBottom: "1px solid var(--border)",
        display: "flex", alignItems: "flex-end", justifyContent: "space-between", gap: 24,
      }}>
        <div style={{ minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 6, flexWrap: "wrap" }}>
            <span className="mono" style={{ fontSize: 11, color: "var(--ink-mute)" }}>
              {run.id} · {run.at}
            </span>
            <StatusBadge status={run.modelStatus || "candidate"} />
            {model && (model.val_spearman != null || model.test_spearman != null) && (
              <span className="mono" style={{
                fontSize: 10.5, color: "var(--ink-soft)",
                padding: "3px 8px", borderRadius: 999,
                background: "var(--card)", border: "1px solid var(--border)",
              }}>
                {model.val_spearman != null ? `val ${model.val_spearman.toFixed(3)}` : ""}
                {model.val_spearman != null && model.test_spearman != null ? " · " : ""}
                {model.test_spearman != null ? `test ${model.test_spearman.toFixed(3)}` : ""}
              </span>
            )}
            {model && model.score_method && model.score_method !== "log_norm" && (
              <span className="mono" style={{
                fontSize: 10.5, color: "var(--accent-ink)",
                padding: "3px 8px", borderRadius: 999,
                background: "var(--accent-soft)", border: "1px solid var(--accent-soft)",
              }}>
                score: {model.score_method}
              </span>
            )}
          </div>
          <h2 className="serif" style={{
            fontSize: 34, lineHeight: 1.1, margin: 0, letterSpacing: "-0.015em",
          }}>
            Top {run.results.length} from <em>{(model && model.theory_tag) || run.model}</em>
          </h2>
          {model && model.description && (
            <div style={{
              marginTop: 6, fontSize: 13, color: "var(--ink-soft)",
              lineHeight: 1.45, maxWidth: 720,
            }}>
              {model.description}
            </div>
          )}
          <div style={{ marginTop: 8, fontSize: 12.5, color: "var(--ink-soft)" }}>
            {run.seeds.toLocaleString()} input tracks · {run.latencyMs} ms · {filtered.length} shown
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <a
            href={api.exportRun(run.id)}
            style={{ textDecoration: "none" }}
          >
            <Button variant="ghost" size="sm" leading={<IconExternal />}>Export JSON</Button>
          </a>
          <Button variant="default" size="sm" leading={<IconRefresh />} onClick={onRerun} disabled={running}>
            Re-run
          </Button>
        </div>
      </div>

      <div style={{
        padding: "14px 40px", display: "flex", alignItems: "center", gap: 14, flexWrap: "wrap",
        background: "var(--bg)", borderBottom: "1px solid var(--border)",
        position: "sticky", top: 0, zIndex: 5,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--ink-mute)" }}>
          <IconFilter /> <span style={{ fontSize: 12 }}>Filters</span>
        </div>

        <SearchBox
          value={filters.query}
          onChange={(q) => setFilters(f => ({ ...f, query: q }))}
        />

        {anyGenre && (
          <Chip
            active={filters.genre !== "all"}
            onClick={() => {
              const idx = availableGenres.indexOf(filters.genre);
              const next = filters.genre === "all"
                ? availableGenres[0]
                : (idx + 1 >= availableGenres.length ? "all" : availableGenres[idx + 1]);
              setFilters(f => ({ ...f, genre: next }));
            }}
          >
            {filters.genre === "all" ? "Any genre" : filters.genre}
          </Chip>
        )}

        {anyYear && (
          <Chip
            active={filters.year !== "all"}
            onClick={() => setFilters(f => ({
              ...f,
              year: f.year === "all" ? "since2020" : f.year === "since2020" ? "since2015" : "all",
            }))}
          >
            {filters.year === "all" ? "Any year" : filters.year === "since2020" ? "Since 2020" : "Since 2015"}
          </Chip>
        )}

        <div style={{ flex: 1 }} />

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 12, color: "var(--ink-mute)" }}>
            <span style={{ color: "var(--good)" }}>↑{up}</span>
            {" · "}
            <span style={{ color: "var(--bad)" }}>↓{down}</span>
          </span>
          <CardStyleToggle value={cardStyle} onChange={onCardStyleChange} />
        </div>
      </div>

      <div style={{ padding: "22px 40px 64px", overflowY: "auto" }}>
        {filtered.length === 0 ? (
          <div style={{
            padding: 48, textAlign: "center", color: "var(--ink-mute)", fontSize: 13,
            background: "var(--card)", borderRadius: 14, border: "1px dashed var(--border-strong)",
          }}>
            No tracks match these filters.
          </div>
        ) : cardStyle === "grid" ? (
          <GridView items={filtered} onVote={onVote} />
        ) : cardStyle === "detailed" ? (
          <DetailedView items={filtered} onVote={onVote} />
        ) : (
          <ListView items={filtered} onVote={onVote} />
        )}
      </div>
    </main>
  );
}

// Carousel header: prev/next, model identity, position dots. Sits at the very
// top of the results pane when the user is comparing models within a family.
// Filled dot = cached, ringed dot = current slot, hollow dot = not yet loaded.
function CompareCarousel({ compare }) {
  const { modelIds, idx, onIdx, cache, loading, models } = compare;
  const currentMid = modelIds[idx];
  const currentModel = models.find(m => m.id === currentMid) || null;
  const canPrev = idx > 0;
  const canNext = idx < modelIds.length - 1;
  const isLoading = !!loading[currentMid];

  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 14,
      padding: "12px 40px",
      borderBottom: "1px solid var(--border)",
      background: "var(--bg-sunken)",
    }}>
      <button
        onClick={() => canPrev && onIdx(idx - 1)}
        disabled={!canPrev}
        title="Previous model (←)"
        style={carouselArrowStyle(canPrev)}
      >‹</button>
      <button
        onClick={() => canNext && onIdx(idx + 1)}
        disabled={!canNext}
        title="Next model (→)"
        style={carouselArrowStyle(canNext)}
      >›</button>

      <div style={{ minWidth: 0, flex: 1, display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
        <span style={{
          fontSize: 14, fontWeight: 600, color: "var(--ink)",
          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          maxWidth: 320,
        }}>
          {currentModel ? (currentModel.theory_tag || currentModel.name) : currentMid}
        </span>
        <span className="mono" style={{ fontSize: 11, color: "var(--ink-mute)" }}>
          {idx + 1} / {modelIds.length}
        </span>
        {currentModel?.val_spearman != null && (
          <span className="mono" style={{
            fontSize: 10.5, color: "var(--ink-soft)",
            padding: "2px 8px", borderRadius: 999,
            background: "var(--card)", border: "1px solid var(--border)",
          }}>
            val {currentModel.val_spearman.toFixed(3)}
          </span>
        )}
        {currentModel?.model_type === "hurdle" && (
          <span className="mono" style={{
            fontSize: 10.5, color: "var(--accent-ink)",
            padding: "2px 8px", borderRadius: 999,
            background: "var(--accent-soft)", border: "1px solid var(--accent-soft)",
          }}>
            hurdle
          </span>
        )}
        {isLoading && (
          <span style={{ fontSize: 11, color: "var(--ink-mute)" }}>
            <Spinner /> running…
          </span>
        )}
      </div>

      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {modelIds.map((mid, i) => {
          const cached = !!cache[mid];
          const active = i === idx;
          const isLoadingDot = !!loading[mid];
          return (
            <button
              key={mid}
              onClick={() => onIdx(i)}
              title={(models.find(m => m.id === mid)?.theory_tag) || mid}
              style={{
                width: 10, height: 10, borderRadius: 999,
                background: active ? "var(--accent)" : cached ? "var(--ink-soft)" : "transparent",
                border: `1.5px solid ${active ? "var(--accent)" : cached ? "var(--ink-soft)" : "var(--border-strong)"}`,
                cursor: "pointer", padding: 0,
                opacity: isLoadingDot && !active ? 0.4 : 1,
                transition: "all 120ms ease",
              }}
            />
          );
        })}
      </div>
    </div>
  );
}

function carouselArrowStyle(enabled) {
  return {
    width: 28, height: 28, borderRadius: 999,
    background: enabled ? "var(--card)" : "transparent",
    border: `1px solid ${enabled ? "var(--border-strong)" : "var(--border)"}`,
    color: enabled ? "var(--ink)" : "var(--ink-mute)",
    cursor: enabled ? "pointer" : "default",
    fontSize: 16, lineHeight: 1, fontFamily: "inherit",
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    transition: "all 120ms ease",
  };
}

function CompareSlotLoading({ compare }) {
  const mid = compare.modelIds[compare.idx];
  const m = compare.models.find(x => x.id === mid);
  return (
    <div style={{ textAlign: "center", color: "var(--ink-soft)" }}>
      <Spinner />
      <div style={{ marginTop: 12, fontSize: 13 }}>
        Loading <em>{(m && (m.theory_tag || m.name)) || mid}</em>…
      </div>
      <div style={{ marginTop: 4, fontSize: 11.5, color: "var(--ink-mute)" }}>
        First flip runs inference (~60s); flips back are instant.
      </div>
    </div>
  );
}

function SearchBox({ value, onChange }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", gap: 6,
      padding: "6px 12px", borderRadius: 999,
      background: "var(--card)", border: "1px solid var(--border)",
      minWidth: 220,
    }}>
      <svg width="13" height="13" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.4" style={{ color: "var(--ink-mute)" }}>
        <circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3 3" strokeLinecap="round"/>
      </svg>
      <input
        type="text"
        placeholder="Search title or artist"
        value={value}
        onChange={e => onChange(e.target.value)}
        style={{
          border: "none", outline: "none", background: "transparent",
          fontFamily: "inherit", fontSize: 12.5, color: "var(--ink)",
          width: 170,
        }}
      />
    </div>
  );
}

function CardStyleToggle({ value, onChange }) {
  const options = [
    { v: "list", label: "List" },
    { v: "grid", label: "Grid" },
    { v: "detailed", label: "Detailed" },
  ];
  return (
    <div style={{
      display: "inline-flex", padding: 3, borderRadius: 999,
      background: "var(--card)", border: "1px solid var(--border)",
    }}>
      {options.map(o => (
        <button
          key={o.v}
          onClick={() => onChange(o.v)}
          style={{
            padding: "5px 12px", borderRadius: 999,
            background: value === o.v ? "var(--ink)" : "transparent",
            color: value === o.v ? "var(--bg)" : "var(--ink-soft)",
            border: "none", cursor: "pointer",
            fontFamily: "inherit", fontSize: 11.5, fontWeight: 500,
            transition: "all 120ms ease",
          }}
        >{o.label}</button>
      ))}
    </div>
  );
}

// ---- List view (default) ----
function ListView({ items, onVote }) {
  const anyMeta = items.some(t => t.genre || t.year);
  const cols = anyMeta
    ? "40px 56px 1fr 160px 120px 100px 140px"
    : "40px 56px 1fr 120px 100px 140px";
  return (
    <div style={{
      background: "var(--card)", borderRadius: 16,
      border: "1px solid var(--border)", overflow: "hidden",
      boxShadow: "var(--shadow-sm)",
    }}>
      <div style={{
        display: "grid",
        gridTemplateColumns: cols,
        gap: 16, padding: "10px 20px",
        borderBottom: "1px solid var(--border)", background: "var(--bg-sunken)",
        fontSize: 10.5, textTransform: "uppercase", letterSpacing: "0.1em",
        color: "var(--ink-mute)", fontWeight: 500,
      }}>
        <div>#</div><div></div><div>Track</div>
        {anyMeta && <div>Genre · Year</div>}
        <div>Score</div><div>Feedback</div>
        <div style={{ textAlign: "right" }}>Play</div>
      </div>
      {items.map((t, i) => (
        <ListRow key={trackKey(t)} track={t} onVote={onVote} even={i % 2 === 0} cols={cols} anyMeta={anyMeta} />
      ))}
    </div>
  );
}

function ListRow({ track, onVote, even, cols, anyMeta }) {
  const hue = hueFor(trackKey(track));
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: cols,
      gap: 16, padding: "12px 20px", alignItems: "center",
      borderBottom: "1px solid var(--border)",
      background: even ? "transparent" : "oklch(0.995 0.004 80)",
      transition: "background 120ms ease",
    }}
    onMouseEnter={(e) => e.currentTarget.style.background = "var(--accent-soft)"}
    onMouseLeave={(e) => e.currentTarget.style.background = even ? "transparent" : "oklch(0.995 0.004 80)"}
    >
      <div className="mono" style={{ fontSize: 12, color: "var(--ink-mute)" }}>
        {String(track.rank).padStart(2, "0")}
      </div>
      <Artwork hue={hue} size={40} title={track.title} radius={8} />
      <div style={{ minWidth: 0 }}>
        <div style={{
          fontSize: 13.5, fontWeight: 500,
          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
        }}>
          {track.title || "—"}
        </div>
        <div style={{
          fontSize: 12, color: "var(--ink-soft)", marginTop: 1,
          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
        }}>
          {track.artist || "—"}
        </div>
        <HurdleBadge track={track} />
      </div>
      {anyMeta && (
        <div className="mono" style={{
          fontSize: 11, color: "var(--ink-mute)",
          whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
        }}>
          {[track.genre, track.year].filter(Boolean).join(" · ") || "—"}
        </div>
      )}
      <ScoreBar value={track.score} width={60} />
      <FeedbackControls vote={track.vote} onVote={(v) => onVote(track, v)} />
      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <PlayButton track={track} />
      </div>
    </div>
  );
}

// ---- Grid view ----
function GridView({ items, onVote }) {
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
      gap: 16,
    }}>
      {items.map(t => (
        <div key={trackKey(t)} style={{
          background: "var(--card)", borderRadius: 16,
          border: "1px solid var(--border)", padding: 14,
          boxShadow: "var(--shadow-sm)", position: "relative",
          transition: "transform 160ms ease, box-shadow 160ms ease",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.transform = "translateY(-2px)"; e.currentTarget.style.boxShadow = "var(--shadow-md)"; }}
        onMouseLeave={(e) => { e.currentTarget.style.transform = ""; e.currentTarget.style.boxShadow = "var(--shadow-sm)"; }}
        >
          <div style={{ position: "relative", marginBottom: 12 }}>
            <Artwork hue={hueFor(trackKey(t))} size={172} title={t.title} radius={12} />
            <div style={{
              position: "absolute", top: 8, left: 8,
              padding: "3px 8px", borderRadius: 999,
              background: "oklch(0.2 0.02 50 / 0.6)", color: "white",
              fontFamily: '"JetBrains Mono", monospace', fontSize: 10.5, fontWeight: 500,
              backdropFilter: "blur(6px)",
            }}>#{t.rank}</div>
          </div>
          <div style={{
            fontSize: 13.5, fontWeight: 600, marginBottom: 2,
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}>
            {t.title || "—"}
          </div>
          <div style={{
            fontSize: 12, color: "var(--ink-soft)", marginBottom: 10,
            whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
          }}>
            {t.artist || "—"}
          </div>
          <HurdleBadge track={t} />
          <div style={{ marginBottom: 10 }}>
            <ScoreBar value={t.score} width={120} />
          </div>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
            <FeedbackControls vote={t.vote} onVote={(v) => onVote(t, v)} compact />
            <PlayButton track={t} compact />
          </div>
        </div>
      ))}
    </div>
  );
}

// ---- Detailed view ----
function DetailedView({ items, onVote }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {items.map(t => (
        <div key={trackKey(t)} style={{
          display: "grid", gridTemplateColumns: "auto 1fr auto",
          gap: 20, alignItems: "center",
          background: "var(--card)", borderRadius: 16,
          border: "1px solid var(--border)", padding: 16,
          boxShadow: "var(--shadow-sm)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
            <div className="mono" style={{ fontSize: 14, color: "var(--ink-mute)", width: 28 }}>
              {String(t.rank).padStart(2, "0")}
            </div>
            <Artwork hue={hueFor(trackKey(t))} size={64} title={t.title} radius={10} />
          </div>
          <div style={{ minWidth: 0 }}>
            <div className="serif" style={{ fontSize: 20, lineHeight: 1.1, letterSpacing: "-0.01em", marginBottom: 4 }}>
              {t.title || "—"}
            </div>
            <div style={{ fontSize: 13, color: "var(--ink-soft)", marginBottom: 10 }}>
              {t.artist || "—"}
              {t.album && (
                <> · <span className="mono" style={{ fontSize: 11.5, color: "var(--ink-mute)" }}>{t.album}</span></>
              )}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 22, flexWrap: "wrap" }}>
              <Stat label="Score">
                <ScoreBar value={t.score} width={80} />
              </Stat>
              {t.p_played != null && (
                <Stat label="P(played)">
                  <span className="mono" style={{ fontSize: 11.5 }}>{t.p_played.toFixed(3)}</span>
                </Stat>
              )}
              {t.score_pred != null && (
                <Stat label="Score | played">
                  <span className="mono" style={{ fontSize: 11.5 }}>{t.score_pred.toFixed(3)}</span>
                </Stat>
              )}
              {t.genre && <Stat label="Genre"><span className="mono" style={{ fontSize: 11.5 }}>{t.genre}</span></Stat>}
              {t.year  && <Stat label="Year"><span className="mono" style={{ fontSize: 11.5 }}>{t.year}</span></Stat>}
              {t.duration != null && (
                <Stat label="Length"><span className="mono" style={{ fontSize: 11.5 }}>{formatDuration(t.duration)}</span></Stat>
              )}
              <Stat label="Embed idx"><span className="mono" style={{ fontSize: 11.5, color: "var(--ink-mute)" }}>{t.embedding_index}</span></Stat>
            </div>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8, alignItems: "flex-end" }}>
            <PlayButton track={t} />
            <FeedbackControls vote={t.vote} onVote={(v) => onVote(t, v)} />
          </div>
        </div>
      ))}
    </div>
  );
}

function Stat({ label, children }) {
  return (
    <div>
      <div style={{ fontSize: 10, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--ink-mute)", marginBottom: 3 }}>
        {label}
      </div>
      {children}
    </div>
  );
}

function formatDuration(s) {
  const m = Math.floor(s / 60); const sec = s % 60;
  return `${m}:${String(sec).padStart(2, "0")}`;
}

function FeedbackControls({ vote, onVote, compact }) {
  const sz = compact ? 12 : 14;
  const pad = compact ? "5px 7px" : "6px 9px";
  return (
    <div style={{ display: "inline-flex", gap: 4 }}>
      <button onClick={() => onVote("up")} title="Good recommendation"
        style={feedbackBtn(vote === "up", "good", pad)}>
        <IconThumbUp size={sz} filled={vote === "up"} />
      </button>
      <button onClick={() => onVote("down")} title="Bad recommendation"
        style={feedbackBtn(vote === "down", "bad", pad)}>
        <IconThumbDown size={sz} filled={vote === "down"} />
      </button>
    </div>
  );
}

function feedbackBtn(active, tone, pad) {
  const color = tone === "good" ? "var(--good)" : "var(--bad)";
  return {
    padding: pad, borderRadius: 8, cursor: "pointer",
    background: active ? (tone === "good" ? "oklch(0.95 0.04 150)" : "oklch(0.96 0.03 25)") : "transparent",
    color: active ? color : "var(--ink-mute)",
    border: `1px solid ${active ? "transparent" : "var(--border)"}`,
    display: "inline-flex", alignItems: "center", justifyContent: "center",
    transition: "all 120ms ease",
    fontFamily: "inherit",
  };
}

function PlayButton({ track, compact }) {
  const term = `${track.artist || ""} ${track.title || ""}`.trim();
  const url = `https://music.apple.com/search?term=${encodeURIComponent(term)}`;
  return (
    <a
      href={url} target="_blank" rel="noopener noreferrer"
      style={{
        display: "inline-flex", alignItems: "center", gap: 7,
        padding: compact ? "6px 10px" : "8px 12px",
        borderRadius: 999, textDecoration: "none",
        background: "var(--ink)", color: "var(--bg)",
        fontSize: compact ? 11.5 : 12, fontWeight: 500,
        transition: "transform 120ms ease, background 120ms ease",
      }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "var(--accent)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "var(--ink)"; }}
    >
      <IconPlay size={10} />
      {compact ? "Play" : "Play on Apple Music"}
    </a>
  );
}

function EmptyState({ onRerun, running }) {
  return (
    <div style={{ textAlign: "center", maxWidth: 420 }}>
      <div style={{
        width: 72, height: 72, borderRadius: 20, margin: "0 auto 18px",
        background: "var(--card)", border: "1px solid var(--border)",
        display: "grid", placeItems: "center",
        boxShadow: "var(--shadow-sm)",
      }}>
        <div style={{
          width: 14, height: 14, borderRadius: 4,
          background: "var(--accent-soft)", border: "1px solid var(--accent)",
        }}/>
      </div>
      <div className="serif" style={{ fontSize: 26, marginBottom: 6, letterSpacing: "-0.01em" }}>
        Ready when you are.
      </div>
      <div style={{ fontSize: 13, color: "var(--ink-soft)", marginBottom: 20, lineHeight: 1.55 }}>
        Pick a model from the left, then hit <em>Run inference</em> to generate
        top-50 recommendations from your library.
      </div>
      <Button variant="accent" size="md" onClick={onRerun} disabled={running} leading={<IconPlay size={11} />}>
        Run inference
      </Button>
    </div>
  );
}

function trackKey(t) {
  return t.embedding_key || `${t.artist}::${t.title}`;
}

// Two-number badge that surfaces both hurdle outputs alongside the ranking
// score, so you can tell whether the row is here because P(played) is high,
// score_pred is high, or both. Renders nothing for non-hurdle results.
function HurdleBadge({ track }) {
  if (track.p_played == null && track.score_pred == null) return null;
  return (
    <div className="mono" style={{
      marginTop: 4, fontSize: 10.5, color: "var(--ink-mute)",
      whiteSpace: "nowrap",
    }}>
      {track.p_played != null && <>p={track.p_played.toFixed(2)}</>}
      {track.p_played != null && track.score_pred != null && " · "}
      {track.score_pred != null && <>s={track.score_pred.toFixed(2)}</>}
    </div>
  );
}

Object.assign(window, { ResultsPanel });
