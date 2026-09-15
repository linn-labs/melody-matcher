// Left-side control panel: Run history, Model selector, Hyperparameters.
// All lists come from the backend — App owns the data and passes it in.

const FAMILY_LABELS = {
  dataset_experiment:  "Dataset experiments",
  negative_experiment: "Negative sampling",
  hyperparam_trial:    "Hyperparam trials",
  production:          "Production",
  other:               "Other",
};
const FAMILY_ORDER = [
  "dataset_experiment",
  "negative_experiment",
  "hyperparam_trial",
  "production",
  "other",
];

function SidePanel({
  models, runHistory,
  currentRunId, onSelectRun,
  modelId, onModelChange,
  onRun, running, canRun,
  familyKey, onFamilyChange,
  onPrewarm, prewarmState,
  prewarmTargetCount,
  rankingMode, onRankingModeChange,
  threshold, onThresholdChange,
  compareEngaged, carouselHurdlePinned, onUnlockHurdle,
}) {
  const model = models.find(m => m.id === modelId);
  const isHurdle = model?.model_type === "hurdle";

  return (
    <aside style={{
      width: 340, flex: "0 0 340px",
      borderRight: "1px solid var(--border)",
      background: "var(--bg-sunken)",
      display: "flex", flexDirection: "column",
      height: "100vh", position: "sticky", top: 0,
      overflow: "hidden",
    }}>
      <div style={{
        padding: "22px 24px 18px",
        borderBottom: "1px solid var(--border)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 8,
            background: "var(--ink)", color: "var(--bg)",
            display: "grid", placeItems: "center",
            fontFamily: '"Instrument Serif", serif', fontSize: 18, lineHeight: 1,
          }}>h</div>
          <div>
            <div style={{ fontSize: 13, fontWeight: 600, lineHeight: 1.1 }}>Harness</div>
            <div className="mono" style={{ fontSize: 10, color: "var(--ink-mute)" }}>melody-matcher · local</div>
          </div>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto", padding: "20px 20px 16px" }}>
        <RunHistoryList
          runs={runHistory}
          currentId={currentRunId}
          onSelect={onSelectRun}
        />

        <div style={{ height: 22 }} />

        <ModelPicker
          models={models}
          modelId={modelId}
          onChange={onModelChange}
          familyKey={familyKey}
          onFamilyChange={onFamilyChange}
        />

        {isHurdle && (
          <>
            <div style={{ height: 18 }} />
            {carouselHurdlePinned ? (
              <HurdlePinnedNotice onUnlock={onUnlockHurdle} />
            ) : (
              <RankingPicker
                rankingMode={rankingMode}
                onRankingModeChange={onRankingModeChange}
                threshold={threshold}
                onThresholdChange={onThresholdChange}
              />
            )}
          </>
        )}

        <div style={{ height: 18 }} />

        {model && <Hyperparams model={model} />}
      </div>

      <div style={{
        padding: 16, borderTop: "1px solid var(--border)",
        background: "var(--bg-sunken)",
      }}>
        <Button
          variant="accent" size="lg"
          onClick={onRun} disabled={running || !canRun}
          leading={running ? <Spinner /> : <IconPlay size={12} />}
          style={{ width: "100%" }}
        >
          {running ? "Running inference…" : "Run inference"}
        </Button>
        <div style={{
          marginTop: 10, fontSize: 11, color: "var(--ink-mute)",
          textAlign: "center",
        }}>
          Scores the full 2.2M-track catalog · top 50 returned
        </div>
        <PrewarmControl
          state={prewarmState}
          targetCount={prewarmTargetCount}
          onPrewarm={onPrewarm}
          disabled={!canRun}
        />
      </div>
    </aside>
  );
}

function PrewarmControl({ state, targetCount, onPrewarm, disabled }) {
  const running = state && state.running;
  const total = state?.total || 0;
  const done = state?.done || 0;
  const pct = total > 0 ? Math.round((done / total) * 100) : 0;

  const lastRunFinished =
    !running && total > 0 && done >= total;

  const label = running
    ? `Pre-warming ${done}/${total}…`
    : lastRunFinished
      ? `Re-warm all ${targetCount}`
      : `Pre-warm all ${targetCount}`;

  return (
    <div style={{ marginTop: 14 }}>
      <Button
        variant="ghost" size="sm"
        onClick={onPrewarm}
        disabled={disabled || running}
        leading={running ? <Spinner /> : null}
        style={{ width: "100%" }}
      >
        {label}
      </Button>
      {running && (
        <div style={{ marginTop: 8 }}>
          <div style={{
            height: 4, background: "var(--border)",
            borderRadius: 999, overflow: "hidden",
          }}>
            <div style={{
              height: "100%", width: `${pct}%`,
              background: "var(--accent)",
              transition: "width 200ms ease",
            }} />
          </div>
          <div className="mono" style={{
            marginTop: 6, fontSize: 10, color: "var(--ink-mute)",
            textAlign: "center", whiteSpace: "nowrap",
            overflow: "hidden", textOverflow: "ellipsis",
          }}>
            {state?.current_name ? `now: ${state.current_name}` : `${pct}%`}
          </div>
        </div>
      )}
      {!running && lastRunFinished && Object.keys(state?.errors || {}).length > 0 && (
        <div style={{
          marginTop: 8, fontSize: 10.5, color: "var(--bad)", textAlign: "center",
        }}>
          {Object.keys(state.errors).length} failed — check logs
        </div>
      )}
      {!running && !lastRunFinished && (
        <div style={{
          marginTop: 6, fontSize: 10, color: "var(--ink-mute)",
          textAlign: "center",
        }}>
          Runs each in series · ~60s each on Mac
        </div>
      )}
    </div>
  );
}

function RunHistoryList({ runs, currentId, onSelect }) {
  const [expanded, setExpanded] = useState(true);
  return (
    <section>
      <SectionHeader
        label="Run history"
        count={runs.length}
        expanded={expanded}
        onToggle={() => setExpanded(v => !v)}
      />
      {expanded && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4, marginTop: 10 }}>
          {runs.length === 0 && (
            <div style={{
              padding: "12px 14px", fontSize: 12, color: "var(--ink-mute)",
              border: "1px dashed var(--border-strong)", borderRadius: 10,
            }}>
              No runs yet. Pick a model and hit Run inference.
            </div>
          )}
          {runs.map((r) => {
            const active = r.id === currentId;
            return (
              <button
                key={r.id}
                onClick={() => onSelect(r)}
                style={{
                  display: "grid", gridTemplateColumns: "1fr auto",
                  gap: 4, padding: "9px 12px",
                  background: active ? "var(--card)" : "transparent",
                  border: `1px solid ${active ? "var(--border-strong)" : "transparent"}`,
                  boxShadow: active ? "var(--shadow-sm)" : "none",
                  borderRadius: 10, cursor: "pointer",
                  fontFamily: "inherit", textAlign: "left",
                  transition: "background 120ms ease",
                }}
                onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "oklch(0.97 0.008 70)"; }}
                onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
              >
                <div style={{ minWidth: 0 }}>
                  <div style={{
                    fontSize: 12.5, fontWeight: 500, color: "var(--ink)",
                    whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                  }}>{r.model}</div>
                  <div className="mono" style={{ fontSize: 10.5, color: "var(--ink-mute)", marginTop: 2 }}>
                    {r.id} · {r.at}
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 10.5, color: "var(--ink-mute)" }}>
                  <span style={{ color: "var(--good)" }}>↑{r.feedback.up}</span>
                  <span style={{ color: "var(--bad)" }}>↓{r.feedback.down}</span>
                </div>
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}

function ModelPicker({ models, modelId, onChange, familyKey, onFamilyChange }) {
  const [expanded, setExpanded] = useState(true);

  const familyCounts = useMemo(() => {
    const c = {};
    for (const m of models) c[m.family_key] = (c[m.family_key] || 0) + 1;
    return c;
  }, [models]);

  const visibleFamilies = FAMILY_ORDER.filter(k => (familyCounts[k] || 0) > 0);

  const filtered = useMemo(() => {
    const list = models.filter(m => m.family_key === familyKey);
    // Sort by val_spearman desc, nulls last.
    return [...list].sort((a, b) => {
      const av = a.val_spearman ?? -Infinity;
      const bv = b.val_spearman ?? -Infinity;
      return bv - av;
    });
  }, [models, familyKey]);

  return (
    <section>
      <SectionHeader
        label="Model"
        count={filtered.length}
        expanded={expanded}
        onToggle={() => setExpanded(v => !v)}
      />
      {expanded && (
        <>
          {visibleFamilies.length > 1 && (
            <div style={{
              marginTop: 10, display: "flex", flexWrap: "wrap", gap: 6,
            }}>
              {visibleFamilies.map(k => {
                const active = k === familyKey;
                return (
                  <button
                    key={k}
                    onClick={() => onFamilyChange(k)}
                    style={{
                      padding: "5px 10px", borderRadius: 999,
                      fontSize: 11, fontWeight: 500,
                      cursor: "pointer", fontFamily: "inherit",
                      background: active ? "var(--ink)" : "var(--card)",
                      color: active ? "var(--bg)" : "var(--ink-soft)",
                      border: `1px solid ${active ? "var(--ink)" : "var(--border)"}`,
                      transition: "all 120ms ease",
                    }}
                  >
                    {FAMILY_LABELS[k] || k}
                    <span className="mono" style={{
                      marginLeft: 6, opacity: 0.7, fontSize: 10,
                    }}>
                      {familyCounts[k]}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
          <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 6 }}>
            {filtered.length === 0 && (
              <div style={{
                padding: "12px 14px", fontSize: 12, color: "var(--ink-mute)",
                border: "1px dashed var(--border-strong)", borderRadius: 10,
              }}>
                No checkpoints in this family.
              </div>
            )}
            {filtered.map(m => {
              const active = m.id === modelId;
              return (
                <button
                  key={m.id}
                  onClick={() => onChange(m.id)}
                  style={{
                    position: "relative",
                    padding: "11px 13px",
                    background: active ? "var(--card)" : "transparent",
                    border: `1px solid ${active ? "var(--border-strong)" : "var(--border)"}`,
                    boxShadow: active ? "var(--shadow-sm)" : "none",
                    borderRadius: 10, cursor: "pointer",
                    fontFamily: "inherit", textAlign: "left",
                    transition: "all 120ms ease",
                  }}
                  onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = "oklch(0.97 0.008 70)"; }}
                  onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = "transparent"; }}
                >
                  <div style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    gap: 8, marginBottom: 4,
                  }}>
                    <div style={{
                      fontSize: 13, fontWeight: 600, minWidth: 0,
                      whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                    }}>{m.theory_tag || m.name}</div>
                    {m.val_spearman != null && (
                      <span className="mono" style={{
                        fontSize: 10.5, color: "var(--ink-mute)",
                        whiteSpace: "nowrap", paddingLeft: 4,
                      }}>
                        val {m.val_spearman.toFixed(3)}
                      </span>
                    )}
                  </div>
                  {m.description && (
                    <div style={{
                      fontSize: 11.5, color: "var(--ink-soft)", lineHeight: 1.4,
                      display: "-webkit-box", WebkitLineClamp: 2,
                      WebkitBoxOrient: "vertical", overflow: "hidden",
                    }}>{m.description}</div>
                  )}
                  <div className="mono" style={{
                    marginTop: 6, fontSize: 10, color: "var(--ink-mute)",
                    whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis",
                  }}>
                    score: {m.score_method || "log_norm"} · trained {m.trained}
                  </div>
                  {active && (
                    <div style={{
                      position: "absolute", top: 10, right: 10,
                      width: 6, height: 6, borderRadius: 999,
                      background: "var(--accent)",
                    }} />
                  )}
                </button>
              );
            })}
          </div>
        </>
      )}
    </section>
  );
}

function RankingPicker({ rankingMode, onRankingModeChange, threshold, onThresholdChange }) {
  const [expanded, setExpanded] = useState(true);
  const opts = [
    { v: "p_played",  label: "P(played)",       hint: "rank by probability of being played" },
    { v: "joint",     label: "Joint",           hint: "P(played) × predicted score" },
    { v: "threshold", label: "Score, gated",    hint: "score when P(played) > threshold" },
  ];
  return (
    <section>
      <SectionHeader
        label="Hurdle ranking"
        count={null}
        expanded={expanded}
        onToggle={() => setExpanded(v => !v)}
      />
      {expanded && (
        <div style={{
          marginTop: 10, padding: 12,
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: 10,
          display: "flex", flexDirection: "column", gap: 6,
        }}>
          {opts.map(o => {
            const active = o.v === rankingMode;
            return (
              <button
                key={o.v}
                onClick={() => onRankingModeChange(o.v)}
                style={{
                  display: "flex", flexDirection: "column",
                  alignItems: "flex-start", gap: 2,
                  padding: "9px 11px", borderRadius: 8,
                  background: active ? "var(--accent-soft)" : "transparent",
                  border: `1px solid ${active ? "var(--accent)" : "var(--border)"}`,
                  cursor: "pointer", fontFamily: "inherit", textAlign: "left",
                  transition: "all 120ms ease",
                }}
              >
                <span style={{
                  fontSize: 12.5, fontWeight: 600,
                  color: active ? "var(--accent-ink)" : "var(--ink)",
                }}>{o.label}</span>
                <span style={{
                  fontSize: 11, color: "var(--ink-soft)",
                }}>{o.hint}</span>
              </button>
            );
          })}
          {rankingMode === "threshold" && (
            <div style={{
              marginTop: 4, padding: "8px 4px 2px",
              borderTop: "1px dashed var(--border)",
            }}>
              <div style={{
                display: "flex", justifyContent: "space-between",
                alignItems: "baseline", marginBottom: 4,
              }}>
                <span style={{ fontSize: 11, color: "var(--ink-mute)" }}>
                  P(played) threshold
                </span>
                <span className="mono" style={{ fontSize: 11.5, color: "var(--ink)" }}>
                  {threshold.toFixed(2)}
                </span>
              </div>
              <input
                type="range" min="0" max="1" step="0.05"
                value={threshold}
                onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
                style={{ width: "100%" }}
              />
            </div>
          )}
        </div>
      )}
    </section>
  );
}


// Shown in place of RankingPicker when the carousel is on a hurdle slot:
// hurdle is pinned to p_played for fair comparison with the scoring models in
// the same family. Unlocking re-runs the slot using the global rankingMode.
function HurdlePinnedNotice({ onUnlock }) {
  return (
    <section>
      <SectionHeader label="Hurdle ranking" count={null} expanded={true} onToggle={() => {}} />
      <div style={{
        marginTop: 10, padding: 12,
        background: "var(--card)",
        border: "1px solid var(--border)",
        borderRadius: 10,
      }}>
        <div style={{ fontSize: 12, color: "var(--ink-soft)", lineHeight: 1.5 }}>
          Pinned to <span className="mono">P(played)</span> for fair comparison with the
          scoring models in this family.
        </div>
        <button
          onClick={onUnlock}
          style={{
            marginTop: 10, padding: "6px 10px",
            background: "transparent", border: "1px solid var(--border)",
            borderRadius: 8, cursor: "pointer", fontFamily: "inherit",
            fontSize: 11.5, color: "var(--ink-soft)",
          }}
        >
          Unlock for this slot
        </button>
      </div>
    </section>
  );
}

function Hyperparams({ model }) {
  const [expanded, setExpanded] = useState(true);
  const entries = Object.entries(model.hyperparams || {});
  return (
    <section>
      <SectionHeader
        label="Hyperparameters"
        count={entries.length}
        expanded={expanded}
        onToggle={() => setExpanded(v => !v)}
      />
      {expanded && (
        <div style={{
          marginTop: 10, padding: 12,
          background: "var(--card)",
          border: "1px solid var(--border)",
          borderRadius: 10,
        }}>
          {model.description && (
            <div style={{ fontSize: 11.5, color: "var(--ink-soft)", lineHeight: 1.5, marginBottom: 10 }}>
              {model.description}
            </div>
          )}
          {entries.length === 0 ? (
            <div style={{ fontSize: 11.5, color: "var(--ink-mute)" }}>
              No hyperparameters saved with this checkpoint.
            </div>
          ) : (
            <div style={{ display: "grid", rowGap: 6 }}>
              {entries.map(([k, v]) => (
                <div key={k} style={{
                  display: "grid", gridTemplateColumns: "1fr auto",
                  gap: 12, alignItems: "baseline",
                }}>
                  <span className="mono" style={{ fontSize: 11, color: "var(--ink-mute)" }}>{k}</span>
                  <span className="mono" style={{
                    fontSize: 11, color: "var(--ink)",
                    maxWidth: 170, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                  }}>{formatVal(v)}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function formatVal(v) {
  if (v == null) return "—";
  if (Array.isArray(v)) return `[${v.join(", ")}]`;
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return String(v);
    return v.toString();
  }
  return String(v);
}

function SectionHeader({ label, count, expanded, onToggle }) {
  return (
    <button
      onClick={onToggle}
      style={{
        display: "flex", alignItems: "center", justifyContent: "space-between",
        width: "100%", padding: 0, background: "transparent", border: "none",
        cursor: "pointer", fontFamily: "inherit", color: "var(--ink-soft)",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{
          fontSize: 11, textTransform: "uppercase", letterSpacing: "0.12em",
          color: "var(--ink-mute)", fontWeight: 500,
        }}>{label}</span>
        {count != null && (
          <span className="mono" style={{ fontSize: 10, color: "var(--ink-mute)" }}>
            {count}
          </span>
        )}
      </div>
      <span style={{ color: "var(--ink-mute)" }}>
        <IconChevron dir={expanded ? "up" : "down"} size={11} />
      </span>
    </button>
  );
}

Object.assign(window, { SidePanel });
