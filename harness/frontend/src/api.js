// Fetch wrappers for the harness backend.

async function apiJSON(url, opts = {}) {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(`${res.status}: ${msg}`);
  }
  return res.json();
}

const api = {
  listModels: () => apiJSON("/api/models"),
  listRuns:   () => apiJSON("/api/runs"),
  getRun:     (id) => apiJSON(`/api/runs/${id}`),
  getLibrary: (id) => apiJSON(`/api/library/${id}`),
  createRun:  (body) => apiJSON("/api/runs", { method: "POST", body: JSON.stringify(body) }),
  feedback:   (runId, body) => apiJSON(`/api/runs/${runId}/feedback`, {
    method: "POST", body: JSON.stringify(body),
  }),
  exportRun: (runId) => `/api/runs/${runId}/export`,
  prewarm: (body) => apiJSON("/api/runs/prewarm", {
    method: "POST", body: JSON.stringify(body),
  }),
  prewarmStatus: () => apiJSON("/api/runs/prewarm/status"),
  importLibrary: async (file, matchPolicy = "aggressive") => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("match_policy", matchPolicy);
    const res = await fetch("/api/library/import", { method: "POST", body: fd });
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch {}
      throw new Error(`${res.status}: ${msg}`);
    }
    return res.json();
  },
};

window.api = api;
