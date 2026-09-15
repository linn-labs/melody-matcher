// App shell — wires library import, model registry, run history, runs, and
// feedback to the harness backend. Persists library_id/modelId/cardStyle in
// localStorage so the app comes back to the same library after a reload.

function App() {
  const [library, setLibrary] = useState(null);       // { id, trackCount, matchedCount, ... }
  const [models, setModels] = useState([]);
  const [modelId, setModelId] = useState(null);
  const [familyKey, setFamilyKey] = useState("dataset_experiment");
  const [runHistory, setRunHistory] = useState([]);
  const [running, setRunning] = useState(false);
  const [loadError, setLoadError] = useState(null);
  const [prewarmState, setPrewarmState] = useState(null);

  const [cardStyle, setCardStyle] = useState(
    (window.__HARNESS_TWEAKS__ && window.__HARNESS_TWEAKS__.cardStyle) || "list"
  );
  // Hurdle-only knobs; ignored by the backend for scoring models.
  const [rankingMode, setRankingMode] = useState("p_played");
  const [threshold, setThreshold] = useState(0.5);

  // -------- Compare-mode state --------
  // Carousel over the active family. compareCache memoizes per-model run
  // results so flipping back to a slot is instant. overrideRun takes priority
  // when set (run-history click, or a manual one-off run on a model outside
  // the active family).
  const [compareCache, setCompareCache] = useState({});         // { [modelId]: Run }
  const [compareIdx, setCompareIdx] = useState(0);
  const [compareLoading, setCompareLoading] = useState({});     // { [modelId]: bool }
  // Hurdle is pinned to p_played in compare mode for fair comparison; users
  // can unlock a hurdle slot to use the global rankingMode/threshold instead.
  const [hurdleUnlocked, setHurdleUnlocked] = useState({});     // { [modelId]: bool }
  const [overrideRun, setOverrideRun] = useState(null);

  // Ordered comparison set: family members sorted by val_spearman desc, same
  // order the side-panel ModelPicker shows. Carousel position N maps to
  // compareModelIds[N].
  const compareModelIds = useMemo(() => {
    const list = models.filter(m => m.family_key === familyKey);
    return [...list].sort((a, b) =>
      (b.val_spearman ?? -Infinity) - (a.val_spearman ?? -Infinity)
    ).map(m => m.id);
  }, [models, familyKey]);

  const compareEnabled = compareModelIds.length >= 2 && !!library;
  const carouselModelId = compareModelIds[compareIdx] ?? null;
  const compareEngaged = compareEnabled && !overrideRun && !!carouselModelId;

  // The run shown in the main panel: explicit override takes priority,
  // otherwise read from the carousel cache.
  const currentRun = overrideRun || (carouselModelId ? compareCache[carouselModelId] : null) || null;

  // -------- Boot: load models + run history, then rehydrate library --------
  useEffect(() => {
    (async () => {
      try {
        const [m, r] = await Promise.all([api.listModels(), api.listRuns()]);
        setModels(m.models || []);
        setRunHistory(r.runs || []);
      } catch (e) {
        setLoadError(e.message || String(e));
      }
    })();
  }, []);

  // Rehydrate from localStorage
  useEffect(() => {
    try {
      const raw = localStorage.getItem("harness.state");
      if (!raw) return;
      const s = JSON.parse(raw);
      if (s.modelId) setModelId(s.modelId);
      if (s.cardStyle) setCardStyle(s.cardStyle);
      if (s.familyKey) setFamilyKey(s.familyKey);
      if (s.rankingMode) setRankingMode(s.rankingMode);
      if (typeof s.threshold === "number") setThreshold(s.threshold);
      if (typeof s.compareIdx === "number") setCompareIdx(s.compareIdx);
      if (s.libraryId) {
        api.getLibrary(s.libraryId).then((lib) => {
          setLibrary({
            id: lib.id,
            display_name: lib.display_name,
            user: lib.user,
            trackCount: lib.trackCount,
            matchedCount: lib.matchedCount,
            topCount: lib.topCount,
            source: lib.source,
          });
        }).catch(() => {
          localStorage.removeItem("harness.state");
        });
      }
    } catch {}
  }, []);

  // Persist
  useEffect(() => {
    try {
      localStorage.setItem("harness.state", JSON.stringify({
        libraryId: library?.id || null,
        modelId,
        cardStyle,
        familyKey,
        rankingMode,
        threshold,
        compareIdx,
      }));
    } catch {}
  }, [library, modelId, cardStyle, familyKey, rankingMode, threshold, compareIdx]);

  // Family change → reset carousel to first slot, clear override.
  useEffect(() => {
    setCompareIdx(0);
    setOverrideRun(null);
  }, [familyKey]);

  // Clamp compareIdx if compareModelIds shrinks.
  useEffect(() => {
    if (compareIdx >= compareModelIds.length && compareModelIds.length > 0) {
      setCompareIdx(0);
    }
  }, [compareModelIds, compareIdx]);

  // Keep sidebar modelId synced with carousel position when engaged, so the
  // side-panel model list highlights the right row and Hyperparams shows the
  // current model.
  useEffect(() => {
    if (compareEngaged && carouselModelId && modelId !== carouselModelId) {
      setModelId(carouselModelId);
    }
  }, [compareEngaged, carouselModelId]);

  // Default modelId when nothing is selected yet (e.g. first boot before
  // compareModelIds has populated, or when the active family is empty).
  useEffect(() => {
    if (modelId != null || models.length === 0) return;
    const inFamily = models.filter(m => m.family_key === familyKey);
    const pool = inFamily.length > 0 ? inFamily : models;
    const sorted = [...pool].sort((a, b) =>
      (b.val_spearman ?? -Infinity) - (a.val_spearman ?? -Infinity)
    );
    setModelId(sorted[0].id);
  }, [models, modelId, familyKey]);

  const refreshRunHistory = async () => {
    try {
      const r = await api.listRuns();
      setRunHistory(r.runs || []);
    } catch {}
  };

  // -------- Compare-slot loader --------
  // Fetches a run for `mid` and stashes it in compareCache. Hurdle models are
  // pinned to p_played unless the user explicitly unlocked that slot.
  // Reads dependencies via refs so callers don't need to thread library/etc.
  const compareCacheRef = useRef(compareCache);
  compareCacheRef.current = compareCache;
  const hurdleUnlockedRef = useRef(hurdleUnlocked);
  hurdleUnlockedRef.current = hurdleUnlocked;
  const modelsRef = useRef(models);
  modelsRef.current = models;
  const libraryRef = useRef(library);
  libraryRef.current = library;
  const rankingModeRef = useRef(rankingMode);
  rankingModeRef.current = rankingMode;
  const thresholdRef = useRef(threshold);
  thresholdRef.current = threshold;

  const loadCompareSlot = async (mid, { force = false } = {}) => {
    if (!mid) return;
    const lib = libraryRef.current;
    if (!lib) return;
    if (!force && compareCacheRef.current[mid]) return;
    const model = modelsRef.current.find(m => m.id === mid);
    if (!model) return;
    const isHurdle = model.model_type === "hurdle";
    const pinned = isHurdle && !hurdleUnlockedRef.current[mid];
    const rmode = pinned ? "p_played" : rankingModeRef.current;
    setCompareLoading(s => ({ ...s, [mid]: true }));
    try {
      const { run } = await api.createRun({
        model_id: mid,
        library_id: lib.id,
        top_k: 50,
        ranking_mode: rmode,
        threshold: thresholdRef.current,
      });
      setCompareCache(s => ({ ...s, [mid]: run }));
      refreshRunHistory();
    } catch (e) {
      setLoadError(e.message || String(e));
    } finally {
      setCompareLoading(s => {
        const next = { ...s };
        delete next[mid];
        return next;
      });
    }
  };

  // Auto-load current slot + prefetch the next slot whenever the carousel
  // position changes, the family changes, or the library becomes available.
  useEffect(() => {
    if (!compareEngaged) return;
    const mid = compareModelIds[compareIdx];
    if (!mid) return;
    if (!compareCache[mid] && !compareLoading[mid]) {
      loadCompareSlot(mid);
    }
    // Prefetch next neighbor.
    const nextMid = compareModelIds[compareIdx + 1];
    if (nextMid && !compareCache[nextMid] && !compareLoading[nextMid]) {
      loadCompareSlot(nextMid);
    }
  }, [compareEngaged, compareIdx, compareModelIds, library]);

  // Keyboard navigation: ←/→ flip carousel. Ignore when an input/textarea is
  // focused so search box typing isn't hijacked.
  useEffect(() => {
    const onKey = (e) => {
      if (!compareEngaged) return;
      const t = e.target;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        setCompareIdx(i => Math.max(0, i - 1));
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        setCompareIdx(i => Math.min(compareModelIds.length - 1, i + 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [compareEngaged, compareModelIds.length]);

  const handleRun = async () => {
    if (!library || !modelId || running) return;
    setRunning(true);
    setLoadError(null);
    try {
      // If we're acting on a model that's part of the compare set, route
      // through loadCompareSlot so the result lands in the cache and the
      // carousel reflects it. Otherwise treat as a one-off override.
      const isCompareModel = compareModelIds.includes(modelId);
      if (isCompareModel) {
        // Force re-fetch so changes to rankingMode/threshold (e.g. after
        // unlocking hurdle) actually re-run.
        await loadCompareSlot(modelId, { force: true });
        setOverrideRun(null);
        const idx = compareModelIds.indexOf(modelId);
        if (idx >= 0) setCompareIdx(idx);
      } else {
        const { run } = await api.createRun({
          model_id: modelId,
          library_id: library.id,
          top_k: 50,
          ranking_mode: rankingMode,
          threshold,
        });
        setOverrideRun(run);
        refreshRunHistory();
      }
    } catch (e) {
      setLoadError(e.message || String(e));
    } finally {
      setRunning(false);
    }
  };

  const handleSelectRun = async (hist) => {
    try {
      const { run } = await api.getRun(hist.id);
      setOverrideRun(run);
      if (run.modelId && models.some(m => m.id === run.modelId)) {
        setModelId(run.modelId);
      }
    } catch (e) {
      setLoadError(e.message || String(e));
    }
  };

  // Sidebar model selection. If user picks a model in the active family, re-
  // engage compare mode at that slot; otherwise just update modelId (the run
  // will be displayed when they hit Run inference).
  const handleModelChange = (newId) => {
    setModelId(newId);
    const idx = compareModelIds.indexOf(newId);
    if (idx >= 0) {
      setOverrideRun(null);
      setCompareIdx(idx);
    }
  };

  const handleUnlockHurdle = () => {
    if (!carouselModelId) return;
    setHurdleUnlocked(s => ({ ...s, [carouselModelId]: true }));
    // Bust the cache entry so the next load picks up the user's rmode/threshold.
    setCompareCache(s => {
      const next = { ...s };
      delete next[carouselModelId];
      return next;
    });
    loadCompareSlot(carouselModelId, { force: true });
  };

  const handleVote = async (track, requested) => {
    if (!currentRun) return;
    const prev = track.vote || null;
    const nextVote = prev === requested ? null : requested;
    const tkey = track.embedding_key || `${track.artist}::${track.title}`;

    // Optimistic update — apply to whichever bucket holds this run.
    const applyVote = (r) => {
      if (!r) return r;
      const results = r.results.map((x) => {
        if ((x.embedding_key || `${x.artist}::${x.title}`) !== tkey) return x;
        const { vote, ...rest } = x;
        return nextVote ? { ...rest, vote: nextVote } : rest;
      });
      const up   = results.filter(x => x.vote === "up").length;
      const down = results.filter(x => x.vote === "down").length;
      return { ...r, results, feedback: { up, down } };
    };

    const runId = currentRun.id;
    if (overrideRun && overrideRun.id === runId) {
      setOverrideRun(applyVote);
    }
    // Also update any cache entry sharing this run id.
    setCompareCache(s => {
      const next = { ...s };
      for (const mid of Object.keys(next)) {
        if (next[mid]?.id === runId) next[mid] = applyVote(next[mid]);
      }
      return next;
    });

    try {
      const counts = await api.feedback(runId, { track_key: tkey, vote: nextVote });
      const applyCounts = (r) => r && r.id === runId ? { ...r, feedback: counts } : r;
      if (overrideRun && overrideRun.id === runId) setOverrideRun(applyCounts);
      setCompareCache(s => {
        const next = { ...s };
        for (const mid of Object.keys(next)) {
          if (next[mid]?.id === runId) next[mid] = applyCounts(next[mid]);
        }
        return next;
      });
      refreshRunHistory();
    } catch (e) {
      setLoadError(e.message || String(e));
    }
  };

  const handleResetLibrary = () => {
    setLibrary(null);
    setOverrideRun(null);
    setCompareCache({});
    setCompareLoading({});
    setHurdleUnlocked({});
  };

  // -------- Pre-warm: kicks the backend to run inference for every model in
  // the currently-active family in series, so subsequent clicks are instant. --
  const prewarmTargets = useMemo(
    () => models.filter(m => m.family_key === familyKey).map(m => m.id),
    [models, familyKey]
  );

  const handlePrewarm = async () => {
    if (!library || prewarmTargets.length === 0) return;
    try {
      await api.prewarm({
        library_id: library.id,
        model_ids: prewarmTargets,
      });
      const s = await api.prewarmStatus();
      setPrewarmState(s);
    } catch (e) {
      setLoadError(e.message || String(e));
    }
  };

  // When prewarm completes, fan out cache-population calls (they'll hit the
  // backend cache and return ~instantly, just turning prewarmed runs into
  // populated carousel slots).
  const prevPrewarmRunning = useRef(false);
  useEffect(() => {
    const running = !!prewarmState?.running;
    const total = prewarmState?.total || 0;
    const done = prewarmState?.done || 0;
    const justFinished = prevPrewarmRunning.current && !running && total > 0 && done >= total;
    prevPrewarmRunning.current = running;
    if (justFinished && libraryRef.current) {
      for (const mid of compareModelIds) {
        if (!compareCacheRef.current[mid] && !compareLoading[mid]) {
          loadCompareSlot(mid);
        }
      }
    }
  }, [prewarmState, compareModelIds]);

  // Poll prewarm status.
  useEffect(() => {
    let cancelled = false;
    let timer = null;
    const tick = async () => {
      try {
        const s = await api.prewarmStatus();
        if (cancelled) return;
        setPrewarmState(s);
        const delay = s.running ? 1500 : 8000;
        timer = setTimeout(tick, delay);
      } catch {
        if (!cancelled) timer = setTimeout(tick, 8000);
      }
    };
    tick();
    return () => { cancelled = true; if (timer) clearTimeout(timer); };
  }, []);

  if (!library) {
    return (
      <>
        {loadError && <GlobalError msg={loadError} onDismiss={() => setLoadError(null)} />}
        <ConnectHero onImport={setLibrary} />
      </>
    );
  }

  const activeModel = models.find(m => m.id === modelId) || null;
  const carouselIsHurdle = compareEngaged && activeModel?.model_type === "hurdle";
  const carouselHurdlePinned = carouselIsHurdle && !hurdleUnlocked[carouselModelId];

  return (
    <div style={{ display: "flex", minHeight: "100vh", background: "var(--bg)" }}>
      <SidePanel
        models={models}
        runHistory={runHistory}
        currentRunId={currentRun?.id}
        onSelectRun={handleSelectRun}
        modelId={modelId}
        onModelChange={handleModelChange}
        onRun={handleRun}
        running={running}
        canRun={!!(library && modelId)}
        familyKey={familyKey}
        onFamilyChange={setFamilyKey}
        onPrewarm={handlePrewarm}
        prewarmState={prewarmState}
        prewarmTargetCount={prewarmTargets.length}
        rankingMode={rankingMode}
        onRankingModeChange={setRankingMode}
        threshold={threshold}
        onThresholdChange={setThreshold}
        compareEngaged={compareEngaged}
        carouselHurdlePinned={carouselHurdlePinned}
        onUnlockHurdle={handleUnlockHurdle}
      />

      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0, position: "relative" }}>
        <div style={{
          position: "absolute", top: 20, right: 40, zIndex: 20,
        }}>
          <LibraryChip library={library} onReset={handleResetLibrary} />
        </div>

        <ResultsPanel
          run={currentRun}
          model={activeModel}
          cardStyle={cardStyle}
          onCardStyleChange={setCardStyle}
          onRerun={handleRun}
          running={running}
          onVote={handleVote}
          compare={compareEngaged ? {
            modelIds: compareModelIds,
            idx: compareIdx,
            onIdx: setCompareIdx,
            cache: compareCache,
            loading: compareLoading,
            models,
          } : null}
        />
      </div>

      {loadError && <GlobalError msg={loadError} onDismiss={() => setLoadError(null)} />}
    </div>
  );
}

function GlobalError({ msg, onDismiss }) {
  return (
    <div style={{
      position: "fixed", bottom: 20, right: 20, zIndex: 200,
      maxWidth: 360, padding: 14, borderRadius: 14,
      background: "var(--card)", border: "1px solid var(--bad)",
      boxShadow: "var(--shadow-lg)",
      animation: "harness-fade 180ms ease",
    }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 10 }}>
        <div style={{ fontSize: 12.5, color: "var(--bad)", fontWeight: 600 }}>Error</div>
        <button onClick={onDismiss} style={{
          background: "transparent", border: "none", cursor: "pointer",
          color: "var(--ink-mute)", padding: 0,
        }}><IconX /></button>
      </div>
      <div style={{ fontSize: 12, color: "var(--ink-soft)", marginTop: 6, whiteSpace: "pre-wrap" }}>
        {msg}
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
