(function () {
  "use strict";

  const LABEL_OPTIONS = [
    ["unmarked", "未标记"],
    ["dead_cell", "死细胞"],
    ["single", "单细胞"],
    ["touching_doublet", "接触双细胞"],
    ["cluster_3plus", "≥3 细胞簇"],
    ["debris", "杂质"],
    ["invalid", "无效"],
    ["uncertain", "不确定"],
  ];
  const BEHAVIOR_LABELS = {
    cell_to_debris: "细胞 → 杂质",
    division_or_growth: "分裂 / 生长",
    stable_debris: "稳定杂质",
    wall_structure_invalid: "孔壁结构 → 无效",
    wall_independent_object: "孔壁独立对象",
    wall_uncertain: "孔壁不确定",
    stable_cell_or_conflict: "稳定细胞 / 冲突",
    decline_without_morphology_evidence: "置信度下降但证据不足",
    no_decisive_temporal_evidence: "无决定性时序证据",
  };
  const DECISION_LABELS = {
    accept_v3: "接受 V3",
    keep_legacy: "保留 V2",
    manual_labels: "人工逐帧标签",
    needs_more: "需要更多证据",
    skip: "跳过",
  };
  const UNIFIED_TRACK_LABELS = [
    ["dead_cell", "死细胞"],
    ["uncertain", "统一不确定"],
    ["debris", "统一杂质"],
    ["invalid", "统一无效"],
    ["unmarked", "不标记"],
  ];
  const FRAME_STATE_LABELS = {
    preserved: "保持",
    degraded: "退化",
    degenerating: "退化中",
    division: "分裂证据",
    uncertain: "不确定",
  };
  const state = {
    runs: [], runId: "", plateId: "", tracks: [], total: 0, page: 1, pageSize: 60,
    selectedTrack: null, overview: null, frameLabels: {}, legacyFrameLabels: {},
    trackLabel: "", unifiedTrack: false, loading: false,
  };

  const $ = (id) => document.getElementById(id);
  const els = {
    run: $("runSelect"), plate: $("plateSelect"), behavior: $("behaviorSelect"),
    focus: $("focusSelect"),
    changed: $("changedOnly"), reviewStatus: $("reviewStatus"), search: $("searchInput"),
    refresh: $("refreshButton"), list: $("trackList"), queueCount: $("queueCount"),
    pageText: $("pageText"), prevPage: $("prevPageButton"), nextPage: $("nextPageButton"),
    empty: $("detailEmpty"), detail: $("detailView"), timepoints: $("timepointGrid"),
    tableBody: $("frameTableBody"), reviewer: $("reviewerInput"), notes: $("notesInput"),
    saveStatus: $("saveStatus"), detailTitle: $("detailTitle"), detailBehavior: $("detailBehavior"),
    detailReviewBadge: $("detailReviewBadge"), detailMeta: $("detailMeta"), detailReason: $("detailReason"),
    prevTrack: $("prevTrackButton"), nextTrack: $("nextTrackButton"),
    plateSummary: $("plateSummary"), healthDot: $("healthDot"), healthText: $("healthText"),
    unifiedBox: $("unifiedConclusionBox"), trackLabel: $("trackLabelSelect"), unifiedHint: $("unifiedConclusionHint"),
    manualSave: $("manualSaveButton"),
  };

  function escapeHtml(value) {
    return String(value == null ? "" : value).replace(/[&<>'"]/g, (char) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
    }[char]));
  }

  function formatNumber(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? number.toFixed(digits) : "—";
  }

  function formatPercent(value) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "—";
  }

  function labelText(value) {
    const found = LABEL_OPTIONS.find(([key]) => key === value);
    return found ? found[1] : (value || "—");
  }

  function behaviorText(value) {
    return BEHAVIOR_LABELS[value] || value || "未分类";
  }

  function behaviorClass(value) {
    if (value === "cell_to_debris") return "warning";
    if (value === "division_or_growth") return "green";
    if (value === "stable_debris") return "purple";
    if (value.indexOf("wall_") === 0) return value === "wall_independent_object" ? "blue" : "warning";
    if (value === "no_decisive_temporal_evidence") return "gray";
    return "";
  }

  function defaultFrameLabel(frame) {
    const raw = String(frame.v3_label || "").toLowerCase();
    if (LABEL_OPTIONS.some(([key]) => key === raw)) return raw;
    if (raw === "cell" || raw === "cell_single" || raw === "single_cell") return "single";
    if (raw.indexOf("double") >= 0) return "touching_doublet";
    if (raw.indexOf("cluster") >= 0 || raw.indexOf("3+") >= 0) return "cluster_3plus";
    if (raw.indexOf("debris") >= 0) return "debris";
    if (raw.indexOf("invalid") >= 0) return "invalid";
    return "uncertain";
  }

  function imageUrl(frame, plateId) {
    const params = new URLSearchParams({
      run_id: state.runId, plate_id: plateId || (state.selectedTrack && state.selectedTrack.plate_id) || state.plateId, candidate_id: frame.candidate_id,
      size: "240", mask: "true",
    });
    return `/api/shadow/image?${params.toString()}`;
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    let payload = null;
    try { payload = await response.json(); } catch (_) { /* non-json error */ }
    if (!response.ok) {
      const detail = payload && payload.detail ? payload.detail : `${response.status} ${response.statusText}`;
      throw new Error(detail);
    }
    return payload;
  }

  function setHealth(ok, text) {
    els.healthDot.className = `status-dot ${ok ? "status-ok" : "status-error"}`;
    els.healthText.textContent = text;
  }

  async function loadHealth() {
    try {
      const payload = await api("/api/shadow/health");
      setHealth(true, `服务正常 · ${payload.runs} 个 run`);
    } catch (error) {
      setHealth(false, `服务异常 · ${error.message}`);
    }
  }

  function populateRuns() {
    els.run.innerHTML = state.runs.length
      ? state.runs.map((run) => `<option value="${escapeHtml(run.run_id)}">${escapeHtml(run.run_id)} · ${escapeHtml(run.mode)}</option>`).join("")
      : `<option value="">没有 Shadow run</option>`;
    els.run.value = state.runId;
  }

  function populatePlates(plates) {
    const options = [`<option value="">全部 plate</option>`].concat(
      plates.map((plate) => `<option value="${escapeHtml(plate.plate_id)}">${escapeHtml(plate.plate_id)}</option>`)
    );
    els.plate.innerHTML = options.join("");
    els.plate.value = state.plateId;
  }

  async function loadRuns() {
    const runs = await api("/api/shadow/runs");
    state.runs = Array.isArray(runs) ? runs : [];
    if (!state.runId || !state.runs.some((run) => run.run_id === state.runId)) {
      state.runId = state.runs.length ? state.runs[0].run_id : "";
    }
    populateRuns();
    if (state.runId) await loadRun();
    else renderEmptyList("没有发现 report.json，请先运行 Shadow 测试。");
  }

  async function loadRun() {
    state.loading = true;
    try {
      state.overview = await api(`/api/shadow/overview?run_id=${encodeURIComponent(state.runId)}`);
      const run = state.runs.find((item) => item.run_id === state.runId);
      const plates = (state.overview && state.overview.plates) || (run ? run.plates.map((plate_id) => ({ plate_id })) : []);
      if (state.plateId && !plates.some((plate) => plate.plate_id === state.plateId)) state.plateId = "";
      populatePlates(plates);
      renderOverview(state.overview);
      state.page = 1;
      await loadTracks();
    } finally {
      state.loading = false;
    }
  }

  function renderOverview(overview) {
    const counts = overview && overview.behavior_counts ? overview.behavior_counts : {};
    $("statTotal").textContent = overview ? overview.total_tracks : "—";
    $("statReviewed").textContent = overview ? overview.reviewed_tracks : "—";
    $("statChanged").textContent = overview ? overview.changed_tracks : "—";
    $("statCellDebris").textContent = counts.cell_to_debris || 0;
    $("statDivision").textContent = counts.division_or_growth || 0;
    $("statStableDebris").textContent = counts.stable_debris || 0;
    $("statWallObject").textContent = counts.wall_independent_object || 0;
    $("statV2V3Wells").textContent = overview ? overview.v2_v3_mismatch_wells : "—";
    $("statLowMatchWells").textContent = overview ? overview.low_match_wells : "—";
    $("statFocusWells").textContent = overview ? overview.focus_wells : "—";
    const plates = (overview && overview.plates) || [];
    els.plateSummary.innerHTML = plates.map((plate) => `<span class="plate-chip">${escapeHtml(plate.plate_id)} <strong>${plate.track_count}</strong></span>`).join("");
  }

  function currentQuery() {
    const params = new URLSearchParams({
      run_id: state.runId, behavior: els.behavior.value, focus: els.focus.value,
      changed_only: String(els.changed.checked),
      review_status: els.reviewStatus.value, search: els.search.value.trim(), page: String(state.page), page_size: String(state.pageSize),
    });
    if (state.plateId) params.set("plate_id", state.plateId);
    return params.toString();
  }

  async function loadTracks() {
    if (!state.runId) return;
    const payload = await api(`/api/shadow/tracks?${currentQuery()}`);
    state.tracks = payload.tracks || [];
    state.total = Number(payload.total || 0);
    renderTrackList();
    const selectedId = state.selectedTrack && state.selectedTrack.track_id;
    const selectedPlate = state.selectedTrack && state.selectedTrack.plate_id;
    const replacement = state.tracks.find((track) => track.track_id === selectedId && track.plate_id === selectedPlate);
    if (replacement) {
      state.selectedTrack = replacement;
      renderDetail(replacement);
    } else if (state.tracks.length) {
      state.selectedTrack = state.tracks[0];
      renderDetail(state.selectedTrack);
    } else {
      state.selectedTrack = null;
      renderDetail(null);
    }
  }

  function renderEmptyList(message) {
    els.list.innerHTML = `<div class="empty-state compact">${escapeHtml(message)}</div>`;
    els.queueCount.textContent = "0";
    els.pageText.textContent = "—";
  }

  function renderTrackList() {
    els.queueCount.textContent = `${state.total} 条`;
    const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
    els.pageText.textContent = `${state.page} / ${pages}`;
    els.prevPage.disabled = state.page <= 1;
    els.nextPage.disabled = state.page >= pages;
    if (!state.tracks.length) {
      renderEmptyList("当前筛选没有轨迹");
      return;
    }
    els.list.innerHTML = state.tracks.map((track) => {
      const selected = state.selectedTrack && state.selectedTrack.track_id === track.track_id && state.selectedTrack.plate_id === track.plate_id;
      const review = track.reviewed ? `<span class="review-mark">✓ 已审</span>` : `<span>待审</span>`;
      const change = track.v2_v3_mismatch ? `<span class="changed-mark">V2/V3差异</span>` : "";
      const low = track.low_match_confidence ? `<span class="low-match-mark">三帧低置信</span>` : "";
      return `<button class="track-item ${selected ? "selected" : ""} ${track.reviewed ? "reviewed" : ""}" data-track-id="${escapeHtml(track.track_id)}" data-plate-id="${escapeHtml(track.plate_id)}" type="button">
        <div class="track-top"><span class="track-well">${escapeHtml(track.well || "未知孔位")}</span><span class="behavior-badge ${behaviorClass(track.behavior)}">${escapeHtml(behaviorText(track.behavior))}</span></div>
        <div class="track-id">${escapeHtml(track.track_id)} · ${track.frame_count} 帧</div>
        <div class="track-reason">${escapeHtml(track.reason || "无附加理由")}</div>
        <div class="track-bottom"><span>${change}${low}</span><span>${review}</span></div>
      </button>`;
    }).join("");
    els.list.querySelectorAll(".track-item").forEach((button) => {
      button.addEventListener("click", () => {
        const selected = state.tracks.find((track) => track.track_id === button.dataset.trackId && track.plate_id === button.dataset.plateId);
        if (selected) { state.selectedTrack = selected; renderTrackList(); renderDetail(selected); }
      });
    });
  }

  function renderDetail(track) {
    if (!track) {
      els.empty.classList.remove("hidden");
      els.detail.classList.add("hidden");
      return;
    }
    els.empty.classList.add("hidden");
    els.detail.classList.remove("hidden");
    els.detailTitle.textContent = `${track.well || "未知孔位"} · ${track.track_id}`;
    els.detailBehavior.textContent = behaviorText(track.behavior);
    els.detailBehavior.className = `behavior-badge ${behaviorClass(track.behavior)}`;
    els.detailReviewBadge.textContent = track.reviewed ? `已审核 · ${DECISION_LABELS[track.review.decision] || track.review.decision}` : "待审核";
    els.detailReviewBadge.className = `review-badge ${track.reviewed ? "reviewed" : ""}`;
    const focusFlags = [];
    if (track.v2_v3_mismatch) focusFlags.push(`V2/V3差异 ${track.v2_v3_mismatch_count} 帧`);
    if (track.low_match_confidence) focusFlags.push(`三帧匹配低置信：${track.low_match_reasons.join(", ")}`);
    els.detailMeta.textContent = `${track.plate_id} · ${track.frame_count} 帧 · 孔壁来源：${track.wall_origin || "none"} · 分裂区间：${track.division_interval || "—"}${focusFlags.length ? ` · ${focusFlags.join(" · ")}` : ""}`;
    els.detailReason.textContent = track.reason || "无附加理由";
    $("evidenceBehavior").textContent = formatNumber(track.behavior_score);
    $("evidenceIdentity").textContent = formatNumber(track.identity_score);
    $("evidenceStatic").textContent = formatNumber(track.static_similarity);
    $("evidenceShape").textContent = formatNumber(track.shape_similarity);
    $("evidenceMorphology").textContent = formatNumber(track.morphology_change_score);
    $("evidenceDegradation").textContent = track.semantic_degradation ? "是" : "否";
    $("evidenceDivision").textContent = track.division_veto ? "已保护" : "—";
    state.unifiedTrack = Boolean(track.requires_unified_label || track.behavior === "cell_to_debris");
    state.trackLabel = state.unifiedTrack
      ? ((track.review && track.review.track_label) || track.track_conclusion || "dead_cell")
      : "";
    state.frameLabels = {};
    state.legacyFrameLabels = {};
    if (track.review && track.review.frame_labels) {
      Object.assign(state.legacyFrameLabels, track.review.frame_labels);
      if (!state.unifiedTrack) Object.assign(state.frameLabels, track.review.frame_labels);
    }
    if (state.unifiedTrack) {
      track.frames.forEach((frame) => { state.frameLabels[frame.candidate_id] = state.trackLabel; });
    }
    els.unifiedBox.classList.toggle("hidden", !state.unifiedTrack);
    if (state.unifiedTrack) {
      els.trackLabel.innerHTML = UNIFIED_TRACK_LABELS.map(([key, label]) => `<option value="${key}">${label}</option>`).join("");
      els.trackLabel.value = state.trackLabel;
      els.unifiedHint.textContent = track.legacy_review_needs_unification
        ? "已有旧格式审核记录尚未统一；当前按轨迹结论展示，保存后会把全部证据帧写成同一个标签。"
        : "这是一条生物学事件轨迹，T0/T1/T2 作为证据帧，不分别保存为细胞 / 不确定 / 杂质。";
      els.manualSave.textContent = "保存统一轨迹结论";
    } else {
      els.manualSave.textContent = "保存人工逐帧标签";
    }
    renderTimepoints(track);
    renderFrameTable(track);
    els.reviewer.value = track.review && track.review.reviewer ? track.review.reviewer : (localStorage.getItem("cellvision-reviewer") || "local_user");
    els.notes.value = track.review && track.review.notes ? track.review.notes : "";
    els.saveStatus.textContent = track.review ? `上次保存：${new Date(track.review.updated_at).toLocaleString()}` : "";
    els.prevTrack.disabled = trackIndex() <= 0;
    els.nextTrack.disabled = trackIndex() < 0 || trackIndex() >= state.tracks.length - 1;
  }

  function renderTimepoints(track) {
    els.timepoints.innerHTML = ["T0", "T1", "T2"].map((timepoint) => {
      const frames = (track.timepoints && track.timepoints[timepoint]) || [];
      const body = frames.length ? frames.map((frame) => `<div class="frame-card">
        <img src="${imageUrl(frame, track.plate_id)}" alt="${escapeHtml(track.well)} ${timepoint} ${escapeHtml(frame.candidate_id)}" loading="lazy">
        <div class="frame-info"><strong>${escapeHtml(frame.v3_label)}</strong><br>${formatPercent(frame.cell_probability)} cell · ${formatPercent(frame.debris_probability)} debris<br>${escapeHtml(FRAME_STATE_LABELS[frame.v3_frame_state] || frame.v3_frame_state || "保持")}</div>
      </div>`).join("") : `<div class="no-frame">无候选</div>`;
      return `<div class="timepoint-column"><div class="timepoint-label"><strong>${timepoint}</strong><span>${frames.length} 个候选</span></div><div class="frame-gallery">${body}</div></div>`;
    }).join("");
  }

  function probabilityClass(value) {
    const number = Number(value);
    if (number >= .7) return "high";
    if (number >= .4) return "mid";
    return "low";
  }

  function labelSelect(frame) {
    const value = state.frameLabels[frame.candidate_id] || defaultFrameLabel(frame);
    const disabled = state.unifiedTrack ? " disabled" : "";
    const previous = state.unifiedTrack && state.legacyFrameLabels[frame.candidate_id]
      && state.legacyFrameLabels[frame.candidate_id] !== state.trackLabel
      ? `<span class="legacy-review-note">旧审核：${escapeHtml(labelText(state.legacyFrameLabels[frame.candidate_id]))}</span>`
      : "";
    return `<select data-candidate-id="${escapeHtml(frame.candidate_id)}" aria-label="${escapeHtml(frame.candidate_id)} 人工标签"${disabled}>${LABEL_OPTIONS.map(([key, label]) => `<option value="${key}" ${value === key ? "selected" : ""}>${label}</option>`).join("")}</select>${previous}`;
  }

  function renderFrameTable(track) {
    els.tableBody.innerHTML = track.frames.map((frame) => {
      const stateLabel = FRAME_STATE_LABELS[frame.v3_frame_state] || frame.v3_frame_state || "保持";
      const flags = [];
      if (frame.v3_would_change) flags.push("Legacy变化");
      if (frame.v2_final_label !== frame.v3_label) flags.push("V2/V3差异");
      if (track.low_match_confidence) flags.push("三帧低置信");
      if (frame.semantic_degradation) flags.push("语义退化");
      if (frame.wall_overlap > 0.35) flags.push("孔壁邻近");
      return `<tr>
        <td><strong>${escapeHtml(frame.timepoint)}</strong></td>
        <td title="${escapeHtml(frame.candidate_id)}">${escapeHtml(frame.candidate_id.slice(-18))}</td>
        <td><span class="label-chip">${escapeHtml(frame.legacy_final_label || frame.legacy_label)}</span></td>
        <td><span class="label-chip v3">${escapeHtml(frame.v3_label)}</span></td>
        <td><span class="label-chip unified">${state.unifiedTrack ? escapeHtml(labelText(track.track_conclusion || state.trackLabel)) : "—"}</span></td>
        <td class="match-metrics">${formatNumber(frame.v2_temporal_same_object_score)} / ${formatNumber(frame.v2_temporal_static_similarity)} / ${formatNumber(frame.v2_temporal_shape_similarity)}</td>
        <td class="prob-cell ${probabilityClass(frame.cell_probability)}">${formatPercent(frame.cell_probability)}</td>
        <td class="prob-cell ${probabilityClass(frame.debris_probability)}">${formatPercent(frame.debris_probability)}</td>
        <td class="state-cell"><span class="state">${escapeHtml(stateLabel)}</span>${flags.length ? `<br><span class="flag">${escapeHtml(flags.join(" · "))}</span>` : ""}</td>
        <td>${labelSelect(frame)}</td>
      </tr>`;
    }).join("");
    els.tableBody.querySelectorAll("select[data-candidate-id]").forEach((select) => {
      select.addEventListener("change", () => { state.frameLabels[select.dataset.candidateId] = select.value; });
    });
  }

  function trackIndex() {
    if (!state.selectedTrack) return -1;
    return state.tracks.findIndex((track) => track.track_id === state.selectedTrack.track_id && track.plate_id === state.selectedTrack.plate_id);
  }

  function moveTrack(offset) {
    const index = trackIndex();
    const next = index + offset;
    if (next >= 0 && next < state.tracks.length) {
      state.selectedTrack = state.tracks[next];
      renderTrackList();
      renderDetail(state.selectedTrack);
    }
  }

  async function saveReview(decision) {
    if (!state.selectedTrack) return;
    const reviewer = els.reviewer.value.trim() || "local_user";
    localStorage.setItem("cellvision-reviewer", reviewer);
    els.saveStatus.textContent = "保存中…";
    document.querySelectorAll(".decision-buttons .button").forEach((button) => { button.disabled = true; });
    try {
      const savedFrameLabels = { ...state.frameLabels };
      if (state.unifiedTrack) {
        state.selectedTrack.frames.forEach((frame) => { savedFrameLabels[frame.candidate_id] = state.trackLabel; });
      }
      const payload = {
        run_id: state.runId, plate_id: state.selectedTrack.plate_id, track_id: state.selectedTrack.track_id,
        decision, reviewer, notes: els.notes.value.trim(), track_label: state.unifiedTrack ? state.trackLabel : "",
        frame_labels: savedFrameLabels,
      };
      const result = await api("/api/shadow/review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      els.saveStatus.textContent = `已保存 · ${new Date(result.review.updated_at).toLocaleString()}`;
      await loadRun();
    } catch (error) {
      els.saveStatus.textContent = `保存失败：${error.message}`;
      els.saveStatus.classList.add("error-text");
    } finally {
      document.querySelectorAll(".decision-buttons .button").forEach((button) => { button.disabled = false; });
    }
  }

  function bindEvents() {
    els.run.addEventListener("change", async () => { state.runId = els.run.value; state.plateId = ""; state.selectedTrack = null; await loadRun(); });
    els.plate.addEventListener("change", async () => { state.plateId = els.plate.value; state.page = 1; await loadTracks(); });
    els.behavior.addEventListener("change", async () => { state.page = 1; await loadTracks(); });
    els.focus.addEventListener("change", async () => { state.page = 1; await loadTracks(); });
    els.changed.addEventListener("change", async () => { state.page = 1; await loadTracks(); });
    els.reviewStatus.addEventListener("change", async () => { state.page = 1; await loadTracks(); });
    let searchTimer = null;
    els.search.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(async () => { state.page = 1; await loadTracks(); }, 220); });
    els.refresh.addEventListener("click", async () => { await loadHealth(); await loadRuns(); });
    els.prevPage.addEventListener("click", async () => { if (state.page > 1) { state.page -= 1; await loadTracks(); } });
    els.nextPage.addEventListener("click", async () => { if (state.page < Math.ceil(state.total / state.pageSize)) { state.page += 1; await loadTracks(); } });
    els.prevTrack.addEventListener("click", () => moveTrack(-1));
    els.nextTrack.addEventListener("click", () => moveTrack(1));
    els.trackLabel.addEventListener("change", () => {
      state.trackLabel = els.trackLabel.value;
      if (state.selectedTrack && state.unifiedTrack) {
        state.selectedTrack.frames.forEach((frame) => { state.frameLabels[frame.candidate_id] = state.trackLabel; });
        renderFrameTable(state.selectedTrack);
      }
    });
    document.querySelectorAll("[data-decision]").forEach((button) => button.addEventListener("click", () => saveReview(button.dataset.decision)));
    document.addEventListener("keydown", (event) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)) return;
      if (event.key === "ArrowLeft") moveTrack(-1);
      if (event.key === "ArrowRight") moveTrack(1);
      if (event.key === "1") saveReview("accept_v3");
      if (event.key === "2") saveReview("keep_legacy");
      if (event.key === "3") saveReview("manual_labels");
    });
  }

  async function init() {
    bindEvents();
    await loadHealth();
    try { await loadRuns(); } catch (error) { setHealth(false, `加载失败 · ${error.message}`); renderEmptyList(error.message); }
  }

  init();
}());
