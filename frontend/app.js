const state = {
  all: [],
  // The Flask session is always the current run only. Firebase history is never rendered.
  scope: "current",
  currentRunId: null,
  currentRunCount: 0,
  currentRunMode: null,
  currentRunStartedHuman: null,
  mode: "all",
  polymer: "all",
  wavelength: "all",
  search: "",
  firstLoad: true,
  timer: null,
};

const polymerInfo = {
  PE: { name: "Polyethylene", descriptor: "Common packaging and consumer plastic" },
  PP: { name: "Polypropylene", descriptor: "Common containers, fibres and consumer products" },
  PS: { name: "Polystyrene", descriptor: "Common foam and rigid packaging plastic" },
  PET: { name: "Polyethylene terephthalate", descriptor: "Common beverage bottles and polyester fibres" },
  PVC: { name: "Polyvinyl chloride", descriptor: "Common pipes, films and flexible products" },
  ABS: { name: "Acrylonitrile butadiene styrene", descriptor: "Common rigid engineering plastic" },
};

const $ = (id) => document.getElementById(id);

function normalizeResults(payload) {
  const rows = Array.isArray(payload?.readings) ? payload.readings : [];
  return rows.filter((item) => item && typeof item === "object");
}

function sortNewest(items) {
  return [...items].sort((a, b) => Number(b.timestamp || 0) - Number(a.timestamp || 0));
}

function formatDate(ts) {
  const d = new Date(Number(ts) * 1000);
  if (Number.isNaN(d.getTime())) return "Unknown time";
  return d.toLocaleString([], {
    day: "2-digit", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function formatHuman(item) {
  if (item.timestamp_human) return item.timestamp_human;
  return formatDate(item.timestamp);
}

function confidencePct(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(100, n * 100));
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function showToast(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2200);
}

function updateConnection(ok, message) {
  const dot = $("statusDot");
  const text = $("connectionText");
  text.textContent = message;
  dot.classList.toggle("offline", !ok);
}

function scopedItems() {
  if (state.scope === "current") {
    if (!state.currentRunId) return [];
    return state.all.filter(
      (item) => item.analysis_run_id === state.currentRunId
    );
  }
  return state.all;
}

function matches(item) {
  if (state.mode !== "all" && String(item.mode || "").toLowerCase() !== state.mode) return false;
  if (state.polymer !== "all" && String(item.polymer || "") !== state.polymer) return false;

  if (state.wavelength !== "all") {
    const wanted = String(state.wavelength);
    const captured = Array.isArray(item.wavelengths_captured)
      ? item.wavelengths_captured.map(String)
      : [];
    const per = item.per_wavelength || {};
    if (!captured.includes(wanted) && !Object.hasOwn(per, wanted)) return false;
  }

  if (state.search) {
    const haystack = [
      item.session_id,
      item.device_id,
      item.mode,
      item.polymer,
      item.primary_wavelength_nm,
      item.analysis_run_id,
      item.source_filename,
      ...(item.wavelengths_captured || []),
    ]
      .map((v) => String(v ?? ""))
      .join(" ")
      .toLowerCase();
    if (!haystack.includes(state.search)) return false;
  }

  return true;
}

function computeStats(items) {
  const total = items.length;
  const live = items.filter((x) => String(x.mode || "").toLowerCase() === "live").length;
  const folder = items.filter((x) => String(x.mode || "").toLowerCase() === "folder").length;
  const particles = items.reduce(
    (sum, x) => sum + (Number.isFinite(Number(x.particle_count)) ? Number(x.particle_count) : 0),
    0
  );
  const confidences = items.map((x) => Number(x.confidence)).filter(Number.isFinite);
  const avg = confidences.length
    ? confidences.reduce((a, b) => a + b, 0) / confidences.length
    : null;
  const polymers = [...new Set(items.map((x) => x.polymer).filter(Boolean))];

  $("totalTests").textContent = total;
  $("testSplit").textContent = `${live} live · ${folder} local`;
  $("particleTotal").textContent = particles;
  $("avgConfidence").textContent = avg === null ? "—" : `${(avg * 100).toFixed(1)}%`;
  $("polymerKinds").textContent = polymers.length;

  const distribution = {};
  items.forEach((x) => {
    const key = x.polymer || "Unknown";
    distribution[key] = (distribution[key] || 0) + 1;
  });

  const distributionEl = $("polymerDistribution");
  const entries = Object.entries(distribution).sort((a, b) => b[1] - a[1]);
  distributionEl.innerHTML = entries.length
    ? entries.map(([code, count]) => `
      <button class="distribution-chip" type="button" data-polymer="${escapeHtml(code)}">
        <span class="distribution-code">${escapeHtml(code)}</span>
        <span class="distribution-name">${escapeHtml(polymerInfo[code]?.name || "Model class")}</span>
        <b>${count}</b>
      </button>`).join("")
    : `<span class="muted">No readings available for this view.</span>`;

  distributionEl.querySelectorAll(".distribution-chip").forEach((button) => {
    button.addEventListener("click", () => {
      $("polymerFilter").value = button.dataset.polymer;
      state.polymer = button.dataset.polymer;
      renderResults();
    });
  });

  populatePolymerFilter(items);
}

function populatePolymerFilter(items) {
  const select = $("polymerFilter");
  const current = state.polymer;
  const values = [...new Set(items.map((x) => x.polymer).filter(Boolean))].sort();
  select.innerHTML = `<option value="all">All polymers</option>` +
    values.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("");
  if (values.includes(current)) select.value = current;
  else { select.value = "all"; state.polymer = "all"; }
}

function populateWavelengthFilter(items) {
  const select = $("wavelengthFilter");
  const values = [...new Set(items.flatMap((item) =>
    Array.isArray(item.wavelengths_captured) ? item.wavelengths_captured : []
  ))].map(Number).filter(Number.isFinite).sort((a, b) => a - b);

  const current = state.wavelength;
  select.innerHTML = `<option value="all">All wavelengths</option>` +
    values.map((v) => `<option value="${v}">${v} nm</option>`).join("");
  if (values.map(String).includes(current)) select.value = current;
  else { select.value = "all"; state.wavelength = "all"; }
}

function buildProbabilityRows(probs) {
  const entries = Object.entries(probs || {}).sort((a, b) => Number(b[1]) - Number(a[1]));
  if (!entries.length) return `<div class="muted">Probability data unavailable for this record.</div>`;
  return entries.map(([label, value]) => {
    const pct = confidencePct(value);
    return `<div class="prob-row">
      <div class="prob-label">${escapeHtml(label)}</div>
      <div class="prob-track"><div class="prob-fill" style="width:${pct}%"></div></div>
      <div class="prob-value">${pct.toFixed(1)}%</div>
    </div>`;
  }).join("");
}

function buildWavelengthRows(per) {
  const entries = Object.entries(per || {}).sort((a, b) => Number(a[0]) - Number(b[0]));
  if (!entries.length) return `<div class="muted">No per-wavelength inference was stored.</div>`;
  return entries.map(([wl, value]) => {
    const polymer = value?.polymer || "—";
    const conf = confidencePct(value?.confidence);
    return `<div class="wl-row">
      <div><span class="wl-number">${escapeHtml(wl)} nm</span><small>Captured wavelength</small></div>
      <strong>${escapeHtml(polymer)}</strong>
      <b>${conf.toFixed(1)}%</b>
    </div>`;
  }).join("");
}

function buildCard(item) {
  const tpl = $("resultCardTemplate").content.cloneNode(true);
  const card = tpl.querySelector(".result-card");
  const polymer = item.polymer || "Unknown";
  card.classList.add(`polymer-${String(polymer).toLowerCase().replace(/[^a-z0-9]+/g, "-")}`);
  const conf = confidencePct(item.confidence);
  const info = polymerInfo[polymer] || {
    name: "Model class",
    descriptor: "Class reported by the current classifier",
  };
  const wavelengths = Array.isArray(item.wavelengths_captured)
    ? [...item.wavelengths_captured].map(Number).filter(Number.isFinite).sort((a, b) => a - b)
    : [];

  card.querySelector(".mode-badge").textContent =
    String(item.mode || "unknown").toUpperCase() === "LIVE" ? "LIVE CAPTURE" : "LOCAL TEST";
  card.querySelector(".mode-badge").classList.toggle("live", String(item.mode || "").toLowerCase() === "live");
  card.querySelector(".timestamp").textContent = formatHuman(item);
  card.querySelector(".polymer-symbol").textContent = polymer;
  card.querySelector(".polymer-name").textContent = polymer;
  card.querySelector(".polymer-full-name").textContent = info.name;
  card.querySelector(".polymer-descriptor").textContent = info.descriptor;
  card.querySelector(".confidence-ring").style.setProperty("--confidence", `${conf}%`);
  card.querySelector(".confidence-value").textContent = `${conf.toFixed(1)}%`;
  card.querySelector(".particles").textContent = Number(item.particle_count || 0);
  card.querySelector(".primary-wl").textContent = item.primary_wavelength_nm ? `${item.primary_wavelength_nm} nm` : "—";
  card.querySelector(".captured-wl").textContent = wavelengths.length ? wavelengths.map((x) => `${x} nm`).join(" · ") : "—";
  card.querySelector(".probability-list").innerHTML = buildProbabilityRows(item.class_probabilities);
  card.querySelector(".wavelength-results").innerHTML = buildWavelengthRows(item.per_wavelength);
  card.querySelector(".session-id").textContent = item.session_id || item._key || "—";
  card.querySelector(".device-id").textContent = item.device_id || "—";
  card.querySelector(".run-id").textContent = item.analysis_run_id || "Legacy / pre-run-tagging record";
  card.querySelector(".result-index").textContent = item.result_index
    ? `${item.result_index}${item.expected_results ? ` of ${item.expected_results}` : ""}`
    : "—";
  card.querySelector(".source-filename").textContent = item.source_filename || "—";
  card.querySelector(".record-key").textContent = item._key || "—";

  const raw = {
    device_id: item.device_id,
    session_id: item.session_id,
    mode: item.mode,
    timestamp: item.timestamp,
    timestamp_human: item.timestamp_human,
    analysis_run_id: item.analysis_run_id,
    run_mode: item.run_mode,
    run_started_timestamp: item.run_started_timestamp,
    run_started_human: item.run_started_human,
    result_index: item.result_index,
    expected_results: item.expected_results,
    source_filename: item.source_filename,
    wavelengths_captured: item.wavelengths_captured,
    primary_wavelength_nm: item.primary_wavelength_nm,
    polymer: item.polymer,
    confidence: item.confidence,
    class_probabilities: item.class_probabilities,
    per_wavelength: item.per_wavelength,
    particle_count: item.particle_count,
    image_urls: item.image_urls || {},
    firebase_key: item._key,
  };
  card.querySelector(".raw-json").textContent = JSON.stringify(raw, null, 2);
  return card;
}

function updateScopeUI() {
  document.querySelectorAll("[data-scope]").forEach((button) => {
    button.classList.toggle("active", button.dataset.scope === state.scope);
  });

  const scoped = scopedItems();
  const scopeLabel = "Current run";
  $("runScopeLabel").textContent = scopeLabel;

  if (state.scope === "current") {
    if (state.currentRunId) {
      const runMode = state.currentRunMode ? state.currentRunMode.toUpperCase() : "RUN";
      $("dataSource").textContent = `${scoped.length} result${scoped.length === 1 ? "" : "s"} · ${runMode} · ${state.currentRunStartedHuman || "Latest run"}`;
      $("runInfo").textContent = `Run ID: ${state.currentRunId}`;
    } else {
      $("dataSource").textContent = "Waiting for a new analysis run";
      $("runInfo").textContent = "Run main_demo.py to start a fresh dashboard session.";
    }
  }
}

function renderResults() {
  const grid = $("resultsGrid");
  const base = scopedItems();
  const visible = sortNewest(base.filter(matches));

  updateScopeUI();
  computeStats(base);
  populateWavelengthFilter(base);
  $("visibleCount").textContent = `${visible.length} ${visible.length === 1 ? "result" : "results"}`;
  $("analysisHeading").textContent = "Current analysis";

  grid.innerHTML = "";
  if (!visible.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">◎</div>
        <h3>No results in the current run</h3>
        <p>Start LIVE capture or run LOCAL folder inference with main_demo.py. Successful Firebase pushes will appear here automatically.</p>
      </div>`;
    return;
  }

  const frag = document.createDocumentFragment();
  visible.forEach((item) => frag.appendChild(buildCard(item)));
  grid.appendChild(frag);
}

async function fetchResults(showLoading = false) {
  if (showLoading) $("loadingState")?.classList.add("active");
  try {
    const response = await fetch(`/api/readings?_=${Date.now()}`, {
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    const payload = await response.json();
    if (!response.ok || !payload.ok) throw new Error(payload?.error || `Dashboard API returned ${response.status}`);

    const next = sortNewest(normalizeResults(payload));
    const oldSignature = state.all.map((x) => x._key).join("|");
    const newSignature = next.map((x) => x._key).join("|");

    state.all = next;
    state.currentRunId = payload.current_run_id || null;
    state.currentRunCount = Number(payload.current_run_count || 0);
    state.currentRunMode = payload.current_run_mode || null;
    state.currentRunStartedHuman = payload.current_run_started_human || null;

    updateConnection(true, "Firebase connected");
    $("lastSync").textContent = `Last sync ${new Date().toLocaleTimeString()}`;

    if (state.firstLoad) {
      state.firstLoad = false;
      showToast(state.currentRunId
        ? `${state.currentRunCount} current-run result${state.currentRunCount === 1 ? "" : "s"} loaded`
        : "Firebase connected — no tagged current run yet");
    } else if (oldSignature !== newSignature) {
      showToast("New AquaTrace result received");
    }

    renderResults();
    $("loadingState")?.remove();
  } catch (error) {
    updateConnection(false, "Dashboard data unavailable");
    $("lastSync").textContent = error.message || "Connection error";
    if (state.firstLoad) {
      $("resultsGrid").innerHTML = `
        <div class="error-state">
          <div class="empty-icon">!</div>
          <h3>Could not load the real Firebase readings</h3>
          <p>${escapeHtml(error.message || "Unknown error")}</p>
          <p>Start <code>python flask_dashboard.py</code> from the AquaTrace project root and ensure <code>models/firebase_key.json</code> exists.</p>
          <button type="button" id="retryBtn">Try again</button>
        </div>`;
      $("retryBtn").addEventListener("click", () => fetchResults(true));
    }
  }
}

function setupTabs() {
  document.querySelectorAll(".nav-item").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
      button.classList.add("active");
      const tab = button.dataset.tab;
      $(`${tab}Panel`).classList.add("active");
      $("pageTitle").textContent = tab === "results" ? "Detection Results" : "Citizen Guidance";
      $("sidebar").classList.remove("open");
    });
  });
}

function setupControls() {
  document.querySelectorAll("[data-scope]").forEach((button) => {
    button.addEventListener("click", () => {
      state.scope = "current";
      renderResults();
    });
  });

  document.querySelectorAll(".seg-btn[data-mode]").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".seg-btn[data-mode]").forEach((b) => b.classList.remove("active"));
      button.classList.add("active");
      state.mode = button.dataset.mode;
      renderResults();
    });
  });

  $("polymerFilter").addEventListener("change", (event) => {
    state.polymer = event.target.value;
    renderResults();
  });

  $("wavelengthFilter").addEventListener("change", (event) => {
    state.wavelength = event.target.value;
    renderResults();
  });

  $("searchBox").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    renderResults();
  });

  $("refreshBtn").addEventListener("click", () => {
    fetchResults(true);
    showToast("Refreshing real readings…");
  });

  $("mobileMenu").addEventListener("click", () => $("sidebar").classList.toggle("open"));
}

setupTabs();
setupControls();
fetchResults(true);
state.timer = setInterval(fetchResults, 3000);
