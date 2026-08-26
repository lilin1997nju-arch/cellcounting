const $ = id => document.getElementById(id);
// The review UI is mounted below either /plates/<slug> (legacy) or
// /projects/<project>/plates/<slug>. Keep API and image requests inside that
// board application instead of escaping to the project hub.
const mountedAppBase = document.querySelector('meta[name="review-base-url"]')?.content
  || (window.location.pathname.match(/^(.*\/plates\/[^/]+)/) || [""])[1]
  || "";
const projectBackUrl = document.querySelector('meta[name="project-back-url"]')?.content || "";
const backReviewList = $("backReviewList");
if (backReviewList && projectBackUrl) backReviewList.href = projectBackUrl;
const appUrl = url => {
  if (!url || /^(https?:|data:|blob:)/.test(url)) return url;
  const normalized = url.startsWith("/") ? url : `/${url}`;
  return `${mountedAppBase}${normalized}`;
};
const reviewTimepoints = ["T0", "T1", "T2"];
const lateTimepoints = ["T3", "T4"];
const timepoints = [...reviewTimepoints, ...lateTimepoints];
const wellTypeNames = {
  single_active: "单细胞有活性",
  single_not_divided: "T0-T2未分裂",
  multi_origin: "多细胞来源",
  no_cell_growth: "无明显生长",
  t0_missing_late_cells: "T0缺失但后期出现细胞",
  positive_control: "阳性对照",
  // Keep old artifact keys readable while plates are rebuilt incrementally.
  single_growth_unconfirmed: "T0-T2未分裂",
  missing_t0_or_late_object: "T0缺失但后期出现细胞",
  no_cell: "无明显生长",
  ambiguous: "T0缺失但后期出现细胞"
};
const growthDecisionNames = {
  obvious_growth: "明显生长",
  no_growth: "无明显生长",
  uncertain: "无法判断",
  pending: "待确认"
};
const labelNames = {
  cell: "均为细胞（保留各帧数量）",
  single: "单细胞",
  touching_doublet: "粘连2细胞",
  cluster_3plus: "3+细胞团",
  debris: "杂质/碎片",
  uncertain: "待定",
  invalid: "无关/误检"
};
labelNames.dead_cell = "V3统一死细胞";
const manualVerdictNames = {
  approved: "合格",
  pending: "待定",
  rejected: "排除",
  unclassified: "未判定"
};
const entryFilterParameters = new URLSearchParams(window.location.search);
const finiteFilter = name => {
  const raw = entryFilterParameters.get(name);
  if (raw === null || raw.trim() === "") return null;
  const value = Number(raw);
  return Number.isFinite(value) && value >= 0 ? value : null;
};
const manualVerdictFilterValues = new Set([
  "approved", "pending", "rejected", "unclassified"
]);
const requestedManualVerdict = String(
  entryFilterParameters.get("manual_verdict") || ""
).toLowerCase();
const entryFilters = {
  coverageMin: finiteFilter("coverage_min"),
  debrisMax: finiteFilter("debris_max"),
  day0CellsMin: finiteFilter("day0_cells_min"),
  day0CellsMax: finiteFilter("day0_cells_max"),
  day1CellsMin: finiteFilter("day1_cells_min"),
  day1CellsMax: finiteFilter("day1_cells_max"),
  day2CellsMin: finiteFilter("day2_cells_min"),
  day2CellsMax: finiteFilter("day2_cells_max"),
  manualVerdict: manualVerdictFilterValues.has(requestedManualVerdict)
    ? requestedManualVerdict
    : "all"
};
const plateFilterFields = {
  coverageMin: "plateCoverageMin",
  debrisMax: "plateDebrisMax",
  day0CellsMin: "plateDay0CellsMin",
  day0CellsMax: "plateDay0CellsMax",
  day1CellsMin: "plateDay1CellsMin",
  day1CellsMax: "plateDay1CellsMax",
  day2CellsMin: "plateDay2CellsMin",
  day2CellsMax: "plateDay2CellsMax"
};
function parsePendingPlateQueue() {
  if (entryFilterParameters.get("pending_queue") !== "1") {
    return { active: false, slugs: [], index: -1 };
  }
  try {
    const parsed = JSON.parse(entryFilterParameters.get("pending_plate_queue") || "[]");
    const slugs = Array.isArray(parsed)
      ? parsed.map(value => String(value).trim()).filter(Boolean).slice(0, 500)
      : [];
    const index = Number(entryFilterParameters.get("pending_queue_index"));
    if (!slugs.length || !Number.isInteger(index) || index < 0 || index >= slugs.length) {
      return { active: false, slugs: [], index: -1 };
    }
    return { active: true, slugs, index };
  } catch (_) {
    return { active: false, slugs: [], index: -1 };
  }
}
const pendingPlateQueue = parsePendingPlateQueue();
if (pendingPlateQueue.active) entryFilters.manualVerdict = "pending";
const cellSubtypeLabels = ["single", "touching_doublet", "cluster_3plus"];
const labelColors = {
  single: "#27d79a",
  touching_doublet: "#24c7d9",
  cluster_3plus: "#9b7cff",
  debris: "#ff9f43",
  uncertain: "#ffd166",
  invalid: "#aab3b8"
};
const state = {
  mode: "pending",
  wells: [],
  screeningWells: [],
  detail: null,
  selectedId: null,
  dirty: new Set(),
  undoAction: null,
  busy: false,
  canvases: new Map(),
  views: new Map(),
  v3TrackLabels: new Map(),
  wellLoadedAt: null,
  prefetchedDetails: new Map(),
  prefetchingDetails: new Map(),
  cellCountSaveJobs: new Map(),
  pendingQueueVisited: new Set(),
  pendingQueueAdvancing: false,
  pendingQueueComplete: false
};

async function api(url, options) {
  const response = await fetch(appUrl(url), options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function setMessage(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function renderUndoButton() {
  const button = $("undoButton");
  if (!button) return;
  const available = Boolean(state.undoAction);
  button.disabled = !available || state.busy;
  button.title = available
    ? `撤销 ${state.undoAction.well} 的上一块板结果（Ctrl/Cmd+Z）`
    : "暂无可撤销的已保存结果（Ctrl/Cmd+Z）";
}

async function refreshUndoAction() {
  try {
    const result = await api("/api/quick-review-undo");
    state.undoAction = result.available ? result.action : null;
  } catch (error) {
    state.undoAction = null;
  }
  renderUndoButton();
}

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(0)}%`;
}

function passesEntryFilters(item) {
  const coverage = Number(item.endpoint_coverage_pct);
  const debris = Number(item.day2_debris_count);
  const cells = [0, 1, 2].map(day => Number(item[`day${day}_cell_count`]));
  const manualVerdict = manualVerdictFilterValues.has(String(item.manual_review_decision).toLowerCase())
    ? String(item.manual_review_decision).toLowerCase()
    : "unclassified";
  if (entryFilters.manualVerdict !== "all" && manualVerdict !== entryFilters.manualVerdict) return false;
  if (entryFilters.coverageMin !== null && (!Number.isFinite(coverage) || coverage <= entryFilters.coverageMin)) return false;
  if (entryFilters.debrisMax !== null && (!Number.isFinite(debris) || debris >= entryFilters.debrisMax)) return false;
  for (const day of [0, 1, 2]) {
    const minimum = entryFilters[`day${day}CellsMin`];
    const maximum = entryFilters[`day${day}CellsMax`];
    if (minimum !== null && (!Number.isFinite(cells[day]) || cells[day] < minimum)) return false;
    if (maximum !== null && (!Number.isFinite(cells[day]) || cells[day] > maximum)) return false;
  }
  return true;
}

function renderEntryFilterSummary(matched, total) {
  const conditions = [];
  if (entryFilters.manualVerdict !== "all") {
    conditions.push(`人工判定 = ${manualVerdictNames[entryFilters.manualVerdict]}`);
  }
  if (entryFilters.coverageMin !== null) conditions.push(`末点覆盖率 > ${entryFilters.coverageMin}%`);
  if (entryFilters.debrisMax !== null) conditions.push(`Day2杂质 < ${entryFilters.debrisMax}`);
  for (const day of [0, 1, 2]) {
    const minimum = entryFilters[`day${day}CellsMin`];
    const maximum = entryFilters[`day${day}CellsMax`];
    if (minimum !== null || maximum !== null) {
      conditions.push(`Day${day}细胞 ${minimum ?? "不限"}～${maximum ?? "不限"}`);
    }
  }
  $("entryFilterSummary").textContent = conditions.length
    ? `本板筛选：${conditions.join("；")}（命中 ${matched}/${total} 孔）`
    : `本板筛选：未设置（${total} 孔）`;
}

function initializePlateFilterEditor() {
  Object.entries(plateFilterFields).forEach(([key, id]) => {
    $(id).value = entryFilters[key] ?? "";
  });
  $("plateManualVerdictFilter").value = entryFilters.manualVerdict;
}

function initializePendingQueue() {
  const notice = $("pendingQueueNotice");
  if (!notice || !pendingPlateQueue.active) return;
  notice.hidden = false;
  $("pendingQueueProgress").textContent = `第 ${pendingPlateQueue.index + 1}/${pendingPlateQueue.slugs.length} 块板；Q/W/E 保存后自动进入下一孔`;
  if (projectBackUrl) $("pendingQueueBack").href = projectBackUrl;
}

function advancePendingPlateQueue() {
  if (!pendingPlateQueue.active || state.pendingQueueAdvancing) return false;
  const nextIndex = pendingPlateQueue.index + 1;
  if (nextIndex >= pendingPlateQueue.slugs.length) {
    state.pendingQueueComplete = true;
    const progress = $("pendingQueueProgress");
    if (progress) progress.textContent = "本轮所有待定孔已依次审核";
    return false;
  }
  state.pendingQueueAdvancing = true;
  const projectUrl = new URL(projectBackUrl || "/", window.location.origin);
  if (!projectUrl.pathname.endsWith("/")) projectUrl.pathname += "/";
  const target = new URL(
    `plates/${encodeURIComponent(pendingPlateQueue.slugs[nextIndex])}/`,
    projectUrl
  );
  target.searchParams.set("manual_verdict", "pending");
  target.searchParams.set("mode", "all");
  target.searchParams.set("pending_queue", "1");
  target.searchParams.set("pending_plate_queue", JSON.stringify(pendingPlateQueue.slugs));
  target.searchParams.set("pending_queue_index", String(nextIndex));
  setMessage(`本板待定孔已审核，正在进入第 ${nextIndex + 1} 块板…`);
  window.location.assign(`${target.pathname}${target.search}`);
  return true;
}

function applyPlateFilterFromEditor() {
  const values = Object.fromEntries(Object.entries(plateFilterFields).map(([key, id]) => [
    key,
    $(id).value.trim() === "" ? null : Number($(id).value),
  ]));
  values.manualVerdict = pendingPlateQueue.active
    ? "pending"
    : manualVerdictFilterValues.has($("plateManualVerdictFilter").value)
    ? $("plateManualVerdictFilter").value
    : "all";
  for (const day of [0, 1, 2]) {
    const minimum = values[`day${day}CellsMin`];
    const maximum = values[`day${day}CellsMax`];
    if (minimum !== null && maximum !== null && minimum > maximum) {
      $("plateFilterError").textContent = `Day${day} 细胞数下限不能大于上限。`;
      $("plateFilterError").hidden = false;
      return;
    }
  }
  Object.assign(entryFilters, values);
  $("plateFilterError").hidden = true;
  const url = new URL(window.location.href);
  const queryNames = {
    coverageMin: "coverage_min",
    debrisMax: "debris_max",
    day0CellsMin: "day0_cells_min",
    day0CellsMax: "day0_cells_max",
    day1CellsMin: "day1_cells_min",
    day1CellsMax: "day1_cells_max",
    day2CellsMin: "day2_cells_min",
    day2CellsMax: "day2_cells_max",
    manualVerdict: "manual_verdict"
  };
  Object.entries(queryNames).forEach(([key, parameter]) => {
    if (entryFilters[key] === null || entryFilters[key] === "all") url.searchParams.delete(parameter);
    else url.searchParams.set(parameter, entryFilters[key]);
  });
  window.history.replaceState({}, "", `${url.pathname}${url.search}${url.hash}`);
  loadWells(state.detail?.well || null);
}

function selectedObject() {
  return state.detail?.objects.find(
    object => object.candidate_id === state.selectedId
  ) || null;
}

function nextWellAfter(well) {
  const index = state.wells.findIndex(item => item.well === well);
  if (index < 0 || index + 1 >= state.wells.length) return null;
  return state.wells[index + 1].well;
}

function warmWellImages(detail) {
  for (const imageInfo of Object.values(detail?.images || {})) {
    if (!imageInfo?.available || !imageInfo.url) continue;
    const image = new Image();
    image.src = appUrl(imageInfo.url);
  }
}

function prefetchWell(well) {
  if (
    !well
    || well === state.detail?.well
    || state.prefetchedDetails.has(well)
    || state.prefetchingDetails.has(well)
  ) return;
  const pending = api(`/api/quick-review-well/${well}`)
    .then(detail => {
      state.prefetchedDetails.set(well, detail);
      warmWellImages(detail);
      while (state.prefetchedDetails.size > 2) {
        state.prefetchedDetails.delete(state.prefetchedDetails.keys().next().value);
      }
      return detail;
    })
    .catch(() => null)
    .finally(() => state.prefetchingDetails.delete(well));
  state.prefetchingDetails.set(well, pending);
}

async function refreshStats() {
  const stats = await api("/api/quick-review-stats");
  if (stats.status !== "ready") {
    setMessage("尚未生成可审核的整合判定结果", true);
    return;
  }
  $("roundId").textContent = stats.round_id.replace("integrated-round-", "");
  $("wellProgress").textContent =
    `${stats.completed_well_count} / ${stats.well_count}`;
  $("objectProgress").textContent =
    `${stats.reviewed_object_count} / ${stats.object_count}`;
  $("correctedCount").textContent = stats.corrected_object_count;
}

async function loadWells(preferredWell = null) {
  const search = $("wellSearch").value.trim();
  const loaded = await api("/api/quick-review-wells?mode=all");
  const entryFiltered = loaded.filter(passesEntryFilters);
  renderEntryFilterSummary(entryFiltered.length, loaded.length);
  const queueFiltered = pendingPlateQueue.active
    ? entryFiltered.filter(item => !state.pendingQueueVisited.has(String(item.well).toUpperCase()))
    : entryFiltered;
  let visible = state.mode === "pending"
    ? queueFiltered.filter(item => !item.completed)
    : state.mode === "reviewed"
    ? queueFiltered.filter(item => item.completed)
    : queueFiltered;
  if (search) {
    const needle = search.toUpperCase();
    visible = visible.filter(item => String(item.well).toUpperCase().includes(needle));
  }
  const typeFilter = $("wellTypeFilter").value;
  state.wells = typeFilter === "all"
    ? visible
    : visible.filter(item => item.screening_status === typeFilter);
  renderWellList();
  if (!state.wells.length) {
    state.detail = null;
    state.selectedId = null;
    if (pendingPlateQueue.active && queueFiltered.length === 0 && advancePendingPlateQueue()) return;
    renderEmptyWorkspace();
    return;
  }
  const target = state.wells.find(item => item.well === preferredWell)
    || state.wells[0];
  await loadWell(target.well);
}

function renderWellList() {
  const list = $("wellList");
  list.innerHTML = "";
  for (const item of state.wells) {
    const button = document.createElement("button");
    button.className = "well-row";
    button.dataset.well = item.well;
    button.classList.toggle("active", state.detail?.well === item.well);
    button.innerHTML = `
      <span class="well-name">${item.well}</span>
      <span class="well-summary">
        ${item.cell_count}细胞 · ${item.debris_count}杂质
        · 人工${manualVerdictNames[item.manual_review_decision] || "未判定"}
        ${item.uncertain_count ? ` · ${item.uncertain_count}待定` : ""}
        ${item.temporal_review_count ? ` · ${item.temporal_review_count}时序复核` : ""}
        ${item.v3_cell_to_debris_count ? ` · ${item.v3_cell_to_debris_count} 条V3统一死细胞轨迹` : ""}
      </span>
      <span class="well-type ${item.screening_status}">
        ${wellTypeNames[item.screening_status] || "判定不明确"}
      </span>
      <span class="well-review-count">${item.reviewed_count}/${item.object_count}</span>
    `;
    button.onclick = () => loadWell(item.well);
    list.appendChild(button);
  }
}

function renderEmptyWorkspace() {
  const queueFinished = pendingPlateQueue.active && state.pendingQueueComplete;
  $("wellTitle").textContent = queueFinished ? "待定孔队列已完成" : "没有待审核孔";
  $("timepointGrid").innerHTML = "";
  $("lateTimepointGrid").innerHTML = "";
  $("lateTimepointSection").hidden = true;
  $("currentWellState").textContent = queueFinished ? "全部待定孔已依次审核" : "本组已完成";
  $("currentWellDetail").textContent = queueFinished
    ? "仍保留为待定的孔会继续显示在项目统计中；可返回项目查看最新数量"
    : "可切换到“已完成”或“全部”查看";
  $("selectionEditor").classList.add("empty-selection");
  $("selectedTitle").textContent = "当前列表没有孔";
  $("selectedDetail").textContent = "";
  renderWellVerdict(null);
  if (queueFinished) setMessage("本轮所有待定孔已依次审核，可返回项目查看最新统计");
}

function renderWellVerdict(decision = state.detail?.screening?.review_decision || "unclassified") {
  const normalized = decision && manualVerdictNames[decision] ? decision : "unclassified";
  $("wellVerdictLabel").textContent = decision === null ? "—" : manualVerdictNames[normalized];
  document.querySelectorAll("[data-well-verdict]").forEach(button => {
    button.classList.toggle("active", decision !== null && button.dataset.wellVerdict === normalized);
    button.disabled = decision === null || state.busy;
  });
}

function updateCurrentWellSummary() {
  const detail = state.detail;
  if (!detail) return;
  const reviewed = detail.objects.filter(object => object.reviewed_label).length;
  const reportLabel = wellTypeNames[detail.screening?.screening_status]
    || detail.report?.final_category_label;
  const reportReason = detail.report?.undetermined_reason_label;
  $("currentWellState").textContent =
    reportLabel || (reviewed === detail.objects.length ? "本孔已完成" : `${detail.objects.length} 个目标`);
  $("currentWellDetail").textContent =
    `已保存 ${reviewed}/${detail.objects.length}；本次修改 ${state.dirty.size}`
    + (reportReason ? `；${reportReason}` : "");
}

async function loadWell(well) {
  if (state.busy) return;
  state.busy = true;
  setMessage(`正在载入 ${well} 的三个时间点…`);
  try {
    let detail = state.prefetchedDetails.get(well) || null;
    if (!detail && state.prefetchingDetails.has(well)) {
      detail = await state.prefetchingDetails.get(well);
    }
    state.prefetchedDetails.delete(well);
    state.detail = detail || await api(`/api/quick-review-well/${well}`);
    state.wellLoadedAt = performance.now();
    state.selectedId = null;
    state.dirty.clear();
    state.v3TrackLabels.clear();
    state.canvases.clear();
    state.views.clear();
    renderWellList();
    renderWell();
    prefetchWell(nextWellAfter(well));
    setMessage(`${well} 已载入：点击任一标记进行快速判定`);
  } catch (error) {
    setMessage(`载入失败：${error.message}`, true);
  } finally {
    state.busy = false;
    renderWellVerdict();
    for (const timepoint of reviewTimepoints) renderCellTotalControl(timepoint);
  }
}

function renderWell() {
  const detail = state.detail;
  if (!detail) return;
  $("wellTitle").textContent = detail.well;
  updateCurrentWellSummary();
  renderWellVerdict();
  const grid = $("timepointGrid");
  const lateGrid = $("lateTimepointGrid");
  grid.innerHTML = "";
  lateGrid.innerHTML = "";
  for (const timepoint of reviewTimepoints) {
    renderTimepointCard(timepoint, grid, true);
  }
  let lateAvailable = false;
  for (const timepoint of lateTimepoints) {
    if (detail.images[timepoint]?.available) {
      lateAvailable = true;
      renderTimepointCard(timepoint, lateGrid, false);
    }
  }
  $("lateTimepointSection").hidden = !lateAvailable;
  renderV3WellSummary();
  renderSelection();
}

function renderV3WellSummary() {
  const summary = $("v3WellSummary");
  if (!summary) return;
  const tracks = (state.detail?.v3_tracks || []).filter(track =>
    String(track.label_mode || "") === "unified_track"
    || track.conclusion
    || track.unified_label
    || track.division_rescue
  );
  if (!tracks.length) {
    summary.hidden = true;
    summary.textContent = "";
    return;
  }
  const deadCellCount = tracks.filter(track =>
    track.conclusion === "dead_cell" || track.unified_label === "dead_cell"
  ).length;
  const preview = tracks.slice(0, 5).map(track => {
    const label = track.division_rescue
      ? "跨轨迹分裂补救"
      : track.reviewed_label || track.conclusion || track.unified_label || "unmarked";
    return `${labelNames[label] || label} (${Number(track.frame_count || 3)}帧)`;
  }).join("；");
  const remaining = tracks.length > 5 ? `；另有 ${tracks.length - 5} 条` : "";
  summary.hidden = false;
  summary.textContent = `V3轨迹语义：本孔 ${tracks.length} 条，${deadCellCount} 条为V3统一死细胞。${preview}${remaining}。点击任一轨迹对象可联合复核 T0/T1/T2，并分别保留各帧细胞数量类型。`;
}

function renderTimepointCard(timepoint, container, annotatable) {
  const detail = state.detail;
  const card = $("timepointTemplate").content.firstElementChild.cloneNode(true);
  const objects = annotatable
    ? detail.objects.filter(object => object.timepoint === timepoint)
    : [];
  const hints = [];
  const imageInfo = detail.images[timepoint];
  card.dataset.timepoint = timepoint;
  card.classList.toggle("late-timepoint-card", !annotatable);
  card.querySelector(".timepoint-name").textContent = imageInfo?.display_label || timepoint;
  card.querySelector(".timepoint-count").textContent = annotatable
    ? `${objects.length} 个目标`
    : (growthDecisionNames[imageInfo?.late_growth_decision] || "待确认");
  card.querySelector(".timepoint-breakdown").textContent = annotatable
    ? breakdownText(objects)
    : lateEvidenceText(imageInfo);
  const cellTotalControl = card.querySelector(".cell-total-control");
  cellTotalControl.hidden = !annotatable;
  if (annotatable) {
    cellTotalControl.querySelector(".cell-total-down").onclick = () => changeTimepointCellTotal(timepoint, -1);
    cellTotalControl.querySelector(".cell-total-up").onclick = () => changeTimepointCellTotal(timepoint, 1);
    cellTotalControl.querySelector(".cell-total-auto").onclick = () => saveTimepointCellTotal(timepoint, null);
  }
  const image = card.querySelector(".well-image");
  const canvas = card.querySelector(".object-overlay");
  const loading = card.querySelector(".image-loading");
  const zoomValue = card.querySelector(".zoom-value");
  state.views.set(timepoint, {
    zoom: 1, panX: 0, panY: 0, mode: "navigate", pointerId: null,
    startX: 0, startY: 0, startPanX: 0, startPanY: 0, moved: false
  });
  if (!imageInfo?.available) {
    loading.textContent = "图像不可用";
    card.classList.add("unavailable");
  } else {
    image.alt = `${detail.well} ${timepoint} 整孔图`;
    image.onload = () => {
      loading.hidden = true;
      const existing = state.canvases.get(timepoint);
      state.canvases.set(timepoint, {
        image, canvas, imageInfo, objects, hints, card, zoomValue,
        hiresLoaded: existing?.hiresLoaded || false,
        defaultViewApplied: existing?.defaultViewApplied || false
      });
      fitAndDraw(timepoint);
    };
    image.src = appUrl(imageInfo.url);
    canvas.onpointerdown = event => pointerDown(timepoint, event);
    canvas.onpointermove = event => pointerMove(timepoint, event);
    canvas.onpointerup = event => pointerUp(timepoint, event);
    canvas.onpointercancel = event => pointerCancel(timepoint, event);
    canvas.onwheel = event => wheelZoom(timepoint, event);
    card.querySelector(".zoom-out").onclick = () => zoomAtCenter(timepoint, 1 / 1.5);
    card.querySelector(".zoom-in").onclick = () => zoomAtCenter(timepoint, 1.5);
    card.querySelector(".reset-view").onclick = () => resetView(timepoint);
    if (annotatable) {
      card.querySelector(".add-missed").onclick = () => toggleAddMissed(timepoint);
    } else {
      const actions = card.querySelector(".late-growth-actions");
      actions.hidden = false;
      actions.querySelectorAll("button").forEach(button => {
        button.classList.toggle(
          "active", button.dataset.growth === imageInfo.late_growth_decision
        );
        button.onclick = () => saveLateGrowthDecision(timepoint, button.dataset.growth);
      });
    }
  }
  container.appendChild(card);
  if (annotatable) renderCellTotalControl(timepoint, card);
}

function lateEvidenceText(imageInfo) {
  if (!imageInfo) return "";
  const source = imageInfo.late_growth_source === "human" ? "人工" : "自动";
  const stageNames = {
    dense_growth_region: "密集生长区域",
    anchor_neighbourhood: "细胞邻域",
    expanded_neighbourhood: "扩大邻域",
    whole_well: "整孔检索",
    whole_well_no_anchor: "无锚点整孔检索"
  };
  const stage = stageNames[imageInfo.late_growth_search_stage]
    || imageInfo.late_growth_search_stage || "";
  return `${source}判定${stage ? ` · ${stage}` : ""}`;
}

function breakdownText(objects) {
  const counts = {};
  for (const object of objects) {
    const label = editableDecisionLabel(object);
    counts[label] = (counts[label] || 0) + 1;
  }
  const cell = (counts.single || 0)
    + (counts.touching_doublet || 0)
    + (counts.cluster_3plus || 0);
  const parts = [`细胞 ${cell}`, `杂质 ${counts.debris || 0}`];
  if (counts.uncertain) parts.push(`待定 ${counts.uncertain}`);
  if (counts.invalid) parts.push(`排除 ${counts.invalid}`);
  return parts.join(" · ");
}

function automaticTimepointCellTotal(timepoint) {
  return (state.detail?.objects || [])
    .filter(object => object.timepoint === timepoint)
    .reduce((total, object) => total + ({
      single: 1,
      touching_doublet: 2,
      cluster_3plus: 3
    }[editableDecisionLabel(object)] || 0), 0);
}

function timepointCellTotal(timepoint) {
  const saved = state.detail?.cell_count_totals?.[timepoint];
  const automatic = automaticTimepointCellTotal(timepoint);
  return {
    automatic_count: automatic,
    cell_count: saved?.source === "human" ? Number(saved.cell_count || 0) : automatic,
    source: saved?.source === "human" ? "human" : "automatic"
  };
}

function renderCellTotalControl(timepoint, card = null) {
  const target = card || document.querySelector(`.timepoint-card[data-timepoint="${timepoint}"]`);
  if (!target || !reviewTimepoints.includes(timepoint)) return;
  const total = timepointCellTotal(timepoint);
  target.querySelector(".cell-total-value").textContent = total.cell_count;
  const source = target.querySelector(".cell-total-source");
  source.textContent = total.source === "human" ? "人工" : "自动";
  source.classList.toggle("human", total.source === "human");
  target.querySelector(".cell-total-down").disabled = state.busy || total.cell_count <= 0;
  target.querySelector(".cell-total-up").disabled = state.busy;
  target.querySelector(".cell-total-auto").disabled = state.busy || total.source !== "human";
}

function changeTimepointCellTotal(timepoint, delta) {
  if (!state.detail || state.busy) return;
  const current = timepointCellTotal(timepoint).cell_count;
  saveTimepointCellTotal(timepoint, Math.max(0, current + delta));
}

function applyOptimisticTimepointCellTotal(well, timepoint, cellCount) {
  if (!state.detail || state.detail.well !== well) return;
  const automatic = automaticTimepointCellTotal(timepoint);
  state.detail.cell_count_totals = state.detail.cell_count_totals || {};
  state.detail.cell_count_totals[timepoint] = {
    automatic_count: automatic,
    cell_count: cellCount === null ? automatic : cellCount,
    source: cellCount === null ? "automatic" : "human"
  };
  renderCellTotalControl(timepoint);
}

function cellCountSaveKey(well, timepoint) {
  return `${well}:${timepoint}`;
}

function scheduleTimepointCellTotalSave(well, timepoint, cellCount) {
  const key = cellCountSaveKey(well, timepoint);
  let job = state.cellCountSaveJobs.get(key);
  if (!job) {
    const saved = state.detail?.cell_count_totals?.[timepoint];
    job = {
      well,
      timepoint,
      confirmed: saved?.source === "human" ? Number(saved.cell_count || 0) : null,
      desired: cellCount,
      timer: null,
      inFlight: false,
      promise: null
    };
    state.cellCountSaveJobs.set(key, job);
  }
  job.desired = cellCount;
  if (job.timer) clearTimeout(job.timer);
  job.timer = setTimeout(() => flushTimepointCellTotalSave(key), 350);
}

async function flushTimepointCellTotalSave(key) {
  const job = state.cellCountSaveJobs.get(key);
  if (!job) return;
  if (job.timer) {
    clearTimeout(job.timer);
    job.timer = null;
  }
  if (job.inFlight) return job.promise;
  const persistedValue = job.desired;
  job.inFlight = true;
  job.promise = (async () => {
    try {
      const result = await api("/api/timepoint-cell-count-review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        keepalive: true,
        body: JSON.stringify({
          well: job.well,
          timepoint: job.timepoint,
          cell_count: persistedValue,
          reviewer: "local_user"
        })
      });
      job.confirmed = persistedValue;
      if (result.screening) applyLocalScreeningUpdate(result.screening);
      if (state.detail?.well === job.well) {
        if (result.screening) state.detail.screening = result.screening;
        if (result.report) state.detail.report = result.report;
      }
      if (Object.is(job.desired, persistedValue)) {
        state.cellCountSaveJobs.delete(key);
        if (state.detail?.well === job.well) {
          setMessage(
            persistedValue === null
              ? `${job.well} ${job.timepoint} 已恢复自动细胞总数`
              : `${job.well} ${job.timepoint} 的人工细胞总数已保存为 ${persistedValue}`
          );
        }
      }
    } catch (error) {
      if (Object.is(job.desired, persistedValue)) {
        state.cellCountSaveJobs.delete(key);
        applyOptimisticTimepointCellTotal(job.well, job.timepoint, job.confirmed);
        if (state.detail?.well === job.well) {
          setMessage(`细胞总数保存失败：${error.message}`, true);
        }
      }
    } finally {
      job.inFlight = false;
      job.promise = null;
      if (state.cellCountSaveJobs.get(key) === job && !Object.is(job.desired, persistedValue)) {
        job.timer = setTimeout(() => flushTimepointCellTotalSave(key), 80);
      }
    }
  })();
  return job.promise;
}

async function flushPendingCellCountSaves(well) {
  while (true) {
    const keys = [...state.cellCountSaveJobs.entries()]
      .filter(([, job]) => job.well === well)
      .map(([key]) => key);
    if (!keys.length) return;
    await Promise.all(keys.map(key => flushTimepointCellTotalSave(key)));
  }
}

function saveTimepointCellTotal(timepoint, cellCount) {
  if (!state.detail || state.busy) return;
  const well = state.detail.well;
  scheduleTimepointCellTotalSave(well, timepoint, cellCount);
  applyOptimisticTimepointCellTotal(well, timepoint, cellCount);
  setMessage(
    cellCount === null
      ? `${well} ${timepoint} 已恢复自动，正在后台保存…`
      : `${well} ${timepoint} 已调整为 ${cellCount}，正在后台保存…`
  );
}

function fitAndDraw(timepoint) {
  const entry = state.canvases.get(timepoint);
  if (!entry) return;
  const { image, canvas } = entry;
  const rect = image.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  applyImageTransform(timepoint);
  drawTimepoint(timepoint);
}

function applyImageTransform(timepoint) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  entry.image.style.transform =
    `translate(${view.panX}px, ${view.panY}px) scale(${view.zoom})`;
  entry.zoomValue.textContent = `${Math.round(view.zoom * 100)}%`;
  entry.card.classList.toggle("adding-missed", view.mode === "add");
  entry.card.querySelector(".add-missed").textContent =
    view.mode === "add" ? "点击图中定位" : "＋漏检";
}

function markerGeometry(entry, object) {
  const width = entry.image.clientWidth;
  const height = entry.image.clientHeight;
  const view = state.views.get(object.timepoint) || {
    zoom: 1,
    panX: 0,
    panY: 0
  };
  const scale = Number($("markerScale").value) / 100;
  const x =
    Number(object.x_px) / entry.imageInfo.width_px
    * width * view.zoom + view.panX;
  const y =
    Number(object.y_px) / entry.imageInfo.height_px
    * height * view.zoom + view.panY;
  const naturalRadius =
    Number(object.v2_instance_diameter_px || object.instance_footprint_diameter_px || object.diameter_px || 8)
    / 2 / entry.imageInfo.width_px
    * width * view.zoom;
  const radius = Math.max(4, Math.min(24, naturalRadius * scale));
  return { x, y, radius };
}

function contourGeometry(entry, object) {
  if (!object.v2_contour_json) return [];
  let points;
  try { points = JSON.parse(object.v2_contour_json); } catch (_) { return []; }
  const view = state.views.get(object.timepoint) || { zoom: 1, panX: 0, panY: 0 };
  return points.map(([x, y]) => [
    Number(x) / entry.imageInfo.width_px * entry.image.clientWidth * view.zoom + view.panX,
    Number(y) / entry.imageInfo.height_px * entry.image.clientHeight * view.zoom + view.panY
  ]);
}

function drawTimepoint(timepoint) {
  const entry = state.canvases.get(timepoint);
  if (!entry) return;
  const { canvas, objects, hints } = entry;
  const dpr = window.devicePixelRatio || 1;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, canvas.width / dpr, canvas.height / dpr);
  for (const object of objects) {
    drawObject(ctx, entry, object);
  }
}

function hintGeometry(entry, hint) {
  const view = state.views.get(hint.timepoint) || {
    zoom: 1, panX: 0, panY: 0
  };
  return {
    x: Number(hint.x_px) / entry.imageInfo.width_px
      * entry.image.clientWidth * view.zoom + view.panX,
    y: Number(hint.y_px) / entry.imageInfo.height_px
      * entry.image.clientHeight * view.zoom + view.panY,
    radius: Math.max(9, Math.min(24, 10 * Math.sqrt(view.zoom)))
  };
}

function drawSearchHint(ctx, entry, hint) {
  const { x, y, radius } = hintGeometry(entry, hint);
  if (
    x < -radius || y < -radius
    || x > entry.canvas.clientWidth + radius
    || y > entry.canvas.clientHeight + radius
  ) return;
  ctx.save();
  ctx.strokeStyle = "#ffdf6c";
  ctx.fillStyle = "rgba(30, 35, 38, .68)";
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(x - radius * .55, y);
  ctx.lineTo(x + radius * .55, y);
  ctx.moveTo(x, y - radius * .55);
  ctx.lineTo(x, y + radius * .55);
  ctx.stroke();
  ctx.font = '700 9px "Segoe UI"';
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillStyle = "#ffdf6c";
  ctx.fillText("补搜", x, y - radius - 2);
  ctx.restore();
}

function drawObject(ctx, entry, object) {
  const { x, y, radius } = markerGeometry(entry, object);
  const label = editableDecisionLabel(object);
  const color = labelColors[label] || labelColors.uncertain;
  const selected = object.candidate_id === state.selectedId;
  const view = state.views.get(object.timepoint);
  const markerOpacity = Math.max(
    0,
    Math.min(1, 1 - (Number(view?.zoom || 1) - 1) / 9)
  );
  ctx.save();
  ctx.globalAlpha = markerOpacity;
  ctx.lineWidth = selected ? 3 : 2;
  ctx.strokeStyle = color;
  ctx.fillStyle = `${color}22`;
  ctx.setLineDash(label === "uncertain" ? [5, 4] : []);
  const contour = contourGeometry(entry, object);
  if (contour.length >= 3 && label !== "invalid") {
    ctx.beginPath();
    ctx.moveTo(contour[0][0], contour[0][1]);
    for (let index = 1; index < contour.length; index += 1) {
      ctx.lineTo(contour[index][0], contour[index][1]);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    if (label === "touching_doublet" || label === "cluster_3plus") {
      ctx.font = `700 ${Math.max(9, radius)}px "Segoe UI"`;
      ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillStyle = color;
      ctx.fillText(label === "touching_doublet" ? "2" : "3+", x, y);
    }
  } else if (label === "invalid") {
    ctx.beginPath();
    ctx.moveTo(x - radius, y - radius);
    ctx.lineTo(x + radius, y + radius);
    ctx.moveTo(x + radius, y - radius);
    ctx.lineTo(x - radius, y + radius);
    ctx.stroke();
  } else if (label === "debris") {
    ctx.beginPath();
    ctx.moveTo(x, y - radius);
    ctx.lineTo(x + radius, y);
    ctx.lineTo(x, y + radius);
    ctx.lineTo(x - radius, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  } else if (label === "touching_doublet") {
    // One candidate is one complete group instance.  The former two-circle
    // glyph looked like duplicate detections, so use one enclosing footprint
    // and a multiplicity badge instead.
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.font = `700 ${Math.max(9, radius)}px "Segoe UI"`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillStyle = color;
    ctx.fillText("2", x, y);
  } else {
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    if (label === "cluster_3plus") {
      ctx.font = `700 ${Math.max(9, radius)}px "Segoe UI"`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = color;
      ctx.fillText("3+", x, y);
    }
  }
  if (selected) {
    ctx.setLineDash([]);
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y, radius + 6, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function canvasPoint(entry, event) {
  const rect = entry.canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top
  };
}

function hitTest(timepoint, x, y) {
  const entry = state.canvases.get(timepoint);
  if (!entry) return null;
  let best = null;
  let bestDistance = Infinity;
  for (const object of entry.objects) {
    const geometry = markerGeometry(entry, object);
    const distance = Math.hypot(geometry.x - x, geometry.y - y);
    if (distance <= Math.max(12, geometry.radius + 7) && distance < bestDistance) {
      best = object;
      bestDistance = distance;
    }
  }
  return best;
}

function hitTestHint(timepoint, x, y) {
  const entry = state.canvases.get(timepoint);
  if (!entry) return null;
  let best = null;
  let bestDistance = Infinity;
  for (const hint of entry.hints || []) {
    const geometry = hintGeometry(entry, hint);
    const distance = Math.hypot(geometry.x - x, geometry.y - y);
    if (distance <= geometry.radius + 8 && distance < bestDistance) {
      best = hint;
      bestDistance = distance;
    }
  }
  return best;
}

function focusSearchHint(timepoint, hint) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  const targetZoom = Math.max(view.zoom, 4);
  const baseX = Number(hint.x_px) / entry.imageInfo.width_px
    * entry.image.clientWidth;
  const baseY = Number(hint.y_px) / entry.imageInfo.height_px
    * entry.image.clientHeight;
  view.zoom = targetZoom;
  view.panX = entry.canvas.clientWidth / 2 - baseX * targetZoom;
  view.panY = entry.canvas.clientHeight / 2 - baseY * targetZoom;
  view.mode = "add";
  clampView(timepoint);
  applyImageTransform(timepoint);
  drawTimepoint(timepoint);
  setMessage(
    `${timepoint} 已定位到时序补搜中心；请在高清图中点击真实细胞中心`
  );
}

function selectObject(best) {
  if (!best) return;
  state.selectedId = best.candidate_id;
  drawAll();
  renderSelection();
}

function clampView(timepoint) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  const width = entry.image.clientWidth;
  const height = entry.image.clientHeight;
  view.panX = Math.min(0, Math.max(width * (1 - view.zoom), view.panX));
  view.panY = Math.min(0, Math.max(height * (1 - view.zoom), view.panY));
}

function setZoom(timepoint, nextZoom, anchorX, anchorY) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  const oldZoom = view.zoom;
  const zoom = Math.max(1, Math.min(10, nextZoom));
  view.panX = anchorX - (anchorX - view.panX) * zoom / oldZoom;
  view.panY = anchorY - (anchorY - view.panY) * zoom / oldZoom;
  view.zoom = zoom;
  ensureHighResolution(timepoint);
  clampView(timepoint);
  applyImageTransform(timepoint);
  drawTimepoint(timepoint);
}

function ensureHighResolution(timepoint) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view || view.zoom < 2.5 || entry.hiresLoaded || !entry.imageInfo.hires_url) return;
  entry.hiresLoaded = true;
  entry.image.src = appUrl(entry.imageInfo.hires_url);
}

function zoomAtCenter(timepoint, factor) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  setZoom(
    timepoint,
    view.zoom * factor,
    entry.canvas.clientWidth / 2,
    entry.canvas.clientHeight / 2
  );
}

function wheelZoom(timepoint, event) {
  event.preventDefault();
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  const point = canvasPoint(entry, event);
  setZoom(
    timepoint,
    view.zoom * (event.deltaY < 0 ? 1.25 : 0.8),
    point.x,
    point.y
  );
}

function resetView(timepoint) {
  const view = state.views.get(timepoint);
  if (!view) return;
  Object.assign(view, { zoom: 1, panX: 0, panY: 0, mode: "navigate" });
  applyImageTransform(timepoint);
  drawTimepoint(timepoint);
}

function toggleAddMissed(timepoint) {
  for (const [key, view] of state.views) {
    view.mode = key === timepoint && view.mode !== "add"
      ? "add"
      : "navigate";
    applyImageTransform(key);
  }
  setMessage(
    state.views.get(timepoint).mode === "add"
      ? `${timepoint} 已进入漏检标注模式：放大到目标后点击中心位置`
      : `${timepoint} 已返回浏览/拖动模式`
  );
}

function pointerDown(timepoint, event) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  const point = canvasPoint(entry, event);
  view.pointerId = event.pointerId;
  view.startX = point.x;
  view.startY = point.y;
  view.startPanX = view.panX;
  view.startPanY = view.panY;
  view.moved = false;
  entry.canvas.setPointerCapture(event.pointerId);
}

function pointerMove(timepoint, event) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view || view.pointerId !== event.pointerId) return;
  const point = canvasPoint(entry, event);
  const dx = point.x - view.startX;
  const dy = point.y - view.startY;
  if (Math.hypot(dx, dy) > 4) view.moved = true;
  if (view.mode === "navigate" && view.zoom > 1) {
    view.panX = view.startPanX + dx;
    view.panY = view.startPanY + dy;
    clampView(timepoint);
    applyImageTransform(timepoint);
    drawTimepoint(timepoint);
  }
}

function pointerUp(timepoint, event) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view || view.pointerId !== event.pointerId) return;
  const point = canvasPoint(entry, event);
  if (!view.moved) {
    if (view.mode === "add") {
      addMissedObject(timepoint, point.x, point.y);
      view.mode = "navigate";
      applyImageTransform(timepoint);
    } else {
      const object = hitTest(timepoint, point.x, point.y);
      if (object) selectObject(object);
      else {
        const hint = hitTestHint(timepoint, point.x, point.y);
        if (hint) focusSearchHint(timepoint, hint);
      }
    }
  }
  view.pointerId = null;
  entry.canvas.releasePointerCapture(event.pointerId);
}

function pointerCancel(timepoint, event) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view) return;
  view.pointerId = null;
  if (entry.canvas.hasPointerCapture(event.pointerId)) {
    entry.canvas.releasePointerCapture(event.pointerId);
  }
}

function addMissedObject(timepoint, displayX, displayY) {
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view || !state.detail) return;
  const baseX = (displayX - view.panX) / view.zoom;
  const baseY = (displayY - view.panY) / view.zoom;
  const x = baseX / entry.image.clientWidth * entry.imageInfo.width_px;
  const y = baseY / entry.image.clientHeight * entry.imageInfo.height_px;
  if (
    x < 0 || y < 0
    || x > entry.imageInfo.width_px
    || y > entry.imageInfo.height_px
  ) return;
  const candidateId =
    `manual-new:${state.detail.well}:${timepoint}:`
    + `${Date.now()}:${Math.round(x)}:${Math.round(y)}`;
  const object = {
    candidate_id: candidateId,
    well: state.detail.well,
    timepoint,
    x_px: x,
    y_px: y,
    area_px: Math.PI * 36,
    diameter_px: 12,
    integrated_label: "single",
    current_label: "single",
    reviewed_label: null,
    decision: null,
    integrated_confidence: 1,
    cell_probability: 1,
    debris_probability: 0,
    invalid_probability: 0,
    single_probability: 1,
    touching_doublet_probability: 0,
    cluster_3plus_probability: 0,
    is_new: true,
    is_manual_missed: true
  };
  state.detail.objects.push(object);
  entry.objects.push(object);
  state.selectedId = candidateId;
  state.dirty.add(candidateId);
  drawAll();
  renderSelection();
  renderTimepointHeaders();
  $("currentWellDetail").textContent =
    `已保存 ${state.detail.objects.filter(item => item.reviewed_label).length}`
    + `/${state.detail.objects.length}；本次修改 ${state.dirty.size}`;
  setMessage(
    `${timepoint} 已新增漏检位置；请确认分类和标记直径后保存`
  );
}

function temporalAppearanceText(object) {
  // The legacy patch matcher is only a fallback when the object graph has not
  // formed a usable multi-frame trajectory.  Once V2/V3 has matched two or
  // more timepoints it must not emit a competing "only N frames" message.
  if (Number(object.v2_temporal_candidate_count || 0) >= 2) return "";
  const names = {
    static_pixel_identity_debris: "\u8de8\u65f6\u95f4\u50cf\u7d20\u7a33\u5b9a\uff0c\u503e\u5411\u6742\u8d28",
    changing_shape_or_growth_cell: "\u8de8\u65f6\u95f4\u53d1\u751f\u53d8\u5316\uff0c\u503e\u5411\u7ec6\u80de",
    ambiguous_temporal_appearance: "\u8de8\u65f6\u95f4\u8bc1\u636e\u4ecd\u4e0d\u8db3",
    insufficient_evidence: "\u672a\u627e\u5230\u8db3\u591f\u7684\u5e73\u884c\u5bf9\u7167",
  };
  const text = names[object.temporal_appearance_status] || "";
  return text
    ? `${text} (${Number(object.temporal_appearance_match_count || 0)}\u5f20)`
    : "";
}

function v3TrackFor(object) {
  const trackId = String(object?.v3_track_id || "");
  if (!trackId || !state.detail) return null;
  return (state.detail.v3_tracks || []).find(
    track => String(track.track_id) === trackId
  ) || null;
}

function hasUnifiedV3Track(object) {
  const track = v3TrackFor(object);
  return Boolean(
    track
    && (String(object.v3_label_mode || "") === "unified_track"
      || track.conclusion
      || track.unified_label
      || track.division_rescue)
  );
}

function v3TrackLabelFor(object) {
  const track = v3TrackFor(object);
  if (!track) return "";
  const label = state.v3TrackLabels.get(track.track_id)
    || track.reviewed_label
    || track.conclusion
    || track.unified_label
    || (track.division_rescue ? "cell" : "dead_cell");
  // Older reviews stored a single/doublet/3+ subtype at track level. Treat
  // those values as the semantic cell family so frame multiplicity can vary.
  return cellSubtypeLabels.includes(label) ? "cell" : label;
}

function frameCellSubtype(object) {
  const candidates = [
    state.dirty.has(object.candidate_id) ? object.current_label : "",
    object.reviewed_label,
    object.v3_proposed_label,
    object.integrated_label,
    object.current_label
  ];
  const subtype = candidates.find(label => cellSubtypeLabels.includes(label));
  if (subtype) return subtype;
  const probabilities = [
    ["single", Number(object.single_probability || 0)],
    ["touching_doublet", Number(object.touching_doublet_probability || 0)],
    ["cluster_3plus", Number(object.cluster_3plus_probability || 0)]
  ];
  probabilities.sort((left, right) => right[1] - left[1]);
  return probabilities[0][0];
}

function v3StorageLabel(label, object) {
  if (label === "cell") return frameCellSubtype(object);
  if (label === "dead_cell") return "debris";
  if (label === "unmarked") {
    return object.reviewed_label || object.integrated_label || "uncertain";
  }
  return label;
}

function applyV3TrackLabel(object, label) {
  const track = v3TrackFor(object);
  if (!track || !hasUnifiedV3Track(object)) return;
  state.v3TrackLabels.set(track.track_id, label);
  for (const target of state.detail.objects) {
    if (String(target.v3_track_id || "") !== String(track.track_id)) continue;
    target.current_label = v3StorageLabel(label, target);
    if (label === "unmarked" && !target.reviewed_label && !target.is_new) {
      state.dirty.delete(target.candidate_id);
    } else {
      state.dirty.add(target.candidate_id);
    }
  }
  drawAll();
  renderSelection();
  renderTimepointHeaders();
}

const finalSourceNames = {
  human_track_review: "人工轨迹语义审核",
  human_frame_review: "人工审核",
  v3_unified_track: "V3 统一轨迹决策",
  v3_temporal: "V3 时序决策",
  v2_temporal: "V2 时序决策",
  integrated_model: "识别模型"
};

function effectiveFinalLabel(object) {
  const track = v3TrackFor(object);
  const pendingTrackLabel = track
    ? state.v3TrackLabels.get(track.track_id)
    : "";
  if (pendingTrackLabel && pendingTrackLabel !== "unmarked") {
    return v3StorageLabel(pendingTrackLabel, object);
  }
  if (state.dirty.has(object.candidate_id)) return object.current_label;
  return object.final_label || object.current_label || "uncertain";
}

function editableDecisionLabel(object) {
  return v3StorageLabel(effectiveFinalLabel(object), object);
}

function persistedDecisionLabel(object) {
  return object.final_review_label
    || v3StorageLabel(
      object.final_label
      || object.reviewed_label
      || object.integrated_label
      || object.current_label
      || "uncertain",
      object
    );
}

function temporalEvidenceHtml(object) {
  const evidence = [];
  const present = value => value !== undefined && value !== null && value !== "";
  const frameCount = Math.max(
    Number(object.v3_track_frame_count || 0),
    Number(object.v2_temporal_candidate_count || 0)
  );
  if (frameCount > 0) {
    evidence.push(`<span>轨迹匹配 ${frameCount}/3${frameCount >= 3 ? "（完整）" : ""}</span>`);
  }
  const identity = present(object.v3_identity_score)
    ? object.v3_identity_score : object.v2_temporal_same_object_score;
  const staticScore = present(object.v3_static_similarity)
    ? object.v3_static_similarity : object.v2_temporal_static_similarity_score;
  if (present(identity)) evidence.push(`<span>同一对象 ${pct(identity)}</span>`);
  if (present(staticScore)) evidence.push(`<span>稳定度 ${pct(staticScore)}</span>`);
  if (present(object.v3_shape_similarity)) {
    evidence.push(`<span>形态相似 ${pct(object.v3_shape_similarity)}</span>`);
  }
  if (present(object.v2_wall_overlap)) {
    evidence.push(`<span>孔壁重叠 ${pct(object.v2_wall_overlap)}</span>`);
  }
  const wallCellVeto = Boolean(
    object.v3_wall_cell_veto || object.v2_static_wall_cell_veto
  );
  if (wallCellVeto) {
    const strongFrames = Math.max(
      Number(object.v3_wall_strong_cell_frame_count || 0),
      Number(object.v2_strong_cell_evidence_frame_count || 0)
    );
    evidence.push(
      `<span>强细胞证据 ${strongFrames}/3（否决孔壁误检）</span>`
    );
  }
  if (present(object.v3_morphology_change_score)) {
    evidence.push(`<span>形态变化 ${pct(object.v3_morphology_change_score)}</span>`);
  }
  if (String(object.v3_track_behavior || "") !== "disabled") {
    evidence.push(object.v3_division_veto
      ? `<span>分裂/增长：有${object.v3_division_interval ? `（${object.v3_division_interval}）` : ""}</span>`
      : "<span>分裂/增长：无</span>");
  }
  const legacyAppearance = temporalAppearanceText(object);
  if (legacyAppearance) evidence.push(`<span>${legacyAppearance}</span>`);
  if (
    object.temporal_completion_status
    && object.temporal_completion_status !== "not_evaluated"
  ) {
    const completion = object.temporal_completion_status === "auto_promoted"
      ? "时序自动补回" : "时序补搜待复核";
    evidence.push(`<span>${completion} ${pct(object.temporal_completion_score)}</span>`);
  }
  return evidence.join("");
}

function renderSelection() {
  const editor = $("selectionEditor");
  const object = selectedObject();
  if (!object) {
    editor.classList.add("empty-selection");
    $("selectedTitle").textContent = "点击图中的标记进行判定";
    $("selectedDetail").textContent =
      "绿色为单细胞，青色为粘连双细胞，紫色为多细胞团，橙色为杂质，黄色虚线圆为待定目标。";
    $("finalDecision").hidden = true;
    $("decisionEvidence").hidden = true;
    $("selectedTemporalEvidence").innerHTML = "";
    $("selectedProbabilities").innerHTML = "";
    $("selectedPatch").removeAttribute("src");
    $("diameterControl").hidden = true;
    if ($("v3TrackNotice")) $("v3TrackNotice").hidden = true;
    document.querySelectorAll(".class-buttons button").forEach(
      button => button.classList.remove("active")
    );
    return;
  }
  editor.classList.remove("empty-selection");
  const finalLabel = effectiveFinalLabel(object);
  $("selectedTitle").textContent =
    `${object.timepoint} · ${labelNames[finalLabel] || finalLabel}`;
  $("selectedDetail").textContent =
    `${object.candidate_id} · 位置 (${Math.round(object.x_px)}, ${Math.round(object.y_px)})`
    + ` · 直径 ${Number(object.diameter_px || 0).toFixed(1)} px`
    + (state.dirty.has(object.candidate_id) ? " · 尚未保存" : "");
  $("selectedPatch").src = appUrl(
    `/api/patch?well=${object.well}&timepoint=${object.timepoint}`
    + `&x=${object.x_px}&y=${object.y_px}&size=256`
  );
  const pendingHumanChange = state.dirty.has(object.candidate_id)
    || Boolean(v3TrackFor(object)
      && state.v3TrackLabels.has(v3TrackFor(object).track_id));
  const finalSource = pendingHumanChange
    ? "尚未保存的人工修正"
    : (finalSourceNames[object.final_source] || "模型决策");
  const finalConfidence = pendingHumanChange
    ? ""
    : (Number(object.final_confidence || 0) > 0
      ? ` · 置信 ${pct(object.final_confidence)}` : "");
  const finalNeedsReview = finalLabel === "uncertain"
    || (!pendingHumanChange && object.final_status === "needs_review");
  $("finalDecision").hidden = false;
  $("finalDecision").classList.toggle(
    "needs-review",
    finalNeedsReview
  );
  $("finalDecisionLabel").textContent = `${finalNeedsReview ? "暂定：" : ""}${labelNames[finalLabel] || finalLabel}`;
  $("finalDecisionMeta").textContent = `${finalSource}${finalConfidence}`;
  $("finalDecisionReason").textContent = pendingHumanChange
    ? "这是尚未保存的人工分类；保存后将成为本目标的最终结论。"
    : (object.final_reason_text || "已按当前最高优先级证据生成最终分类。");
  $("decisionEvidence").hidden = false;
  $("selectedTemporalEvidence").innerHTML = temporalEvidenceHtml(object)
    || "<span>没有可用的跨帧证据</span>";
  $("selectedProbabilities").innerHTML = `
    <span>细胞 ${pct(object.cell_probability)}</span>
    <span>杂质 ${pct(object.debris_probability)}</span>
    <span>无效 ${pct(object.invalid_probability)}</span>
    <span>单细胞 ${pct(object.single_probability)}</span>
    <span>双细胞 ${pct(object.touching_doublet_probability)}</span>
    <span>3+ ${pct(object.cluster_3plus_probability)}</span>
  `;
  const v3Notice = $("v3TrackNotice");
  const v3TrackDetail = $("v3TrackDetail");
  const v3TrackLabel = $("v3TrackLabel");
  const track = v3TrackFor(object);
  if (v3Notice && v3TrackDetail && v3TrackLabel && hasUnifiedV3Track(object)) {
    const label = v3TrackLabelFor(object);
    v3Notice.hidden = false;
    v3TrackDetail.textContent = [
      `轨迹 ${track.track_id} · ${track.frame_count} 帧`,
      `V3：${labelNames[label] || labelNames.dead_cell}`,
      track.reason || object.v3_reason || "多帧统一证据",
      track.division_rescue ? "跨轨迹分裂补救已触发" : "",
      label === "cell"
        ? "轨迹级仅确认均为细胞；单/双/3+按各帧判定保留"
        : "保存时将 T0/T1/T2 作为同一语义类别处理"
    ].filter(Boolean).join(" · ");
    v3TrackLabel.value = [
      "dead_cell",
      "cell",
      "uncertain",
      "debris",
      "invalid",
      "unmarked"
    ].includes(label)
      ? label
      : "dead_cell";
    v3TrackLabel.onchange = event => applyV3TrackLabel(object, event.target.value);
  } else if (v3Notice) {
    v3Notice.hidden = true;
    if (v3TrackLabel) v3TrackLabel.onchange = null;
  }
  const canResize = Boolean(object.is_new || object.is_manual_missed);
  $("diameterControl").hidden = !canResize;
  $("diameterValue").textContent =
    `${Number(object.diameter_px || 12).toFixed(0)} px`;
  document.querySelectorAll(".class-buttons button").forEach(button => {
    button.classList.toggle(
      "active",
      button.dataset.label === editableDecisionLabel(object)
    );
  });
}

function changeSelectedDiameter(delta) {
  const object = selectedObject();
  if (!object || !(object.is_new || object.is_manual_missed)) return;
  object.diameter_px = Math.max(
    4,
    Math.min(96, Number(object.diameter_px || 12) + delta)
  );
  object.area_px = Math.PI * (object.diameter_px / 2) ** 2;
  state.dirty.add(object.candidate_id);
  drawAll();
  renderSelection();
}

function setSelectedLabel(label) {
  const object = selectedObject();
  if (!object || state.busy) return;
  if (
    !state.dirty.has(object.candidate_id)
    && label === editableDecisionLabel(object)
  ) return;
  if (
    hasUnifiedV3Track(object)
    && v3TrackLabelFor(object) === "cell"
    && cellSubtypeLabels.includes(label)
  ) {
    // Persist the semantic migration as ``cell`` even when an older review
    // stored ``single``/``touching_doublet`` at track level. The selected
    // frame keeps its own multiplicity below.
    const track = v3TrackFor(object);
    state.v3TrackLabels.set(track.track_id, "cell");
  } else if (hasUnifiedV3Track(object)) {
    applyV3TrackLabel(object, label);
    return;
  }
  object.current_label = label;
  if (
    label === persistedDecisionLabel(object)
    && !object.is_new
  ) {
    state.dirty.delete(object.candidate_id);
  } else {
    state.dirty.add(object.candidate_id);
  }
  drawAll();
  renderSelection();
  renderTimepointHeaders();
  $("currentWellDetail").textContent =
    `已保存 ${state.detail.objects.filter(item => item.reviewed_label).length}`
    + `/${state.detail.objects.length}；本次修改 ${state.dirty.size}`;
}

function renderTimepointHeaders() {
  document.querySelectorAll(".timepoint-card").forEach(card => {
    const timepoint = card.dataset.timepoint;
    if (!reviewTimepoints.includes(timepoint)) return;
    const objects = state.detail.objects.filter(
      object => object.timepoint === timepoint
    );
    card.querySelector(".timepoint-breakdown").textContent =
      breakdownText(objects);
    renderCellTotalControl(timepoint, card);
  });
}

async function saveLateGrowthDecision(timepoint, decision) {
  if (!state.detail || state.busy) return;
  const well = state.detail.well;
  state.busy = true;
  setMessage(`正在保存 ${well} ${timepoint} 生长判定…`);
  try {
    const result = await api("/api/late-growth-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        well, timepoint, decision, reviewer: "local_user"
      })
    });

    const imageInfo = state.detail.images[timepoint];
    if (imageInfo) {
      imageInfo.late_growth_decision = decision;
      imageInfo.late_growth_source = "human";
      imageInfo.late_growth_search_stage = "";
      imageInfo.growth_regions = [];
      imageInfo.growth_overlay_style = "none";
    }
    if (result.report) state.detail.report = result.report;
    applyLocalScreeningUpdate(result.screening);
    updateLateGrowthCard(timepoint);
    updateCurrentWellSummary();
    setMessage(`${well} ${timepoint} 已标记为${growthDecisionNames[decision]}`);
  } catch (error) {
    setMessage(`生长判定保存失败：${error.message}`, true);
  } finally {
    state.busy = false;
  }
}

async function saveManualVerdict(decision) {
  if (!state.detail || state.busy || !manualVerdictNames[decision]) return;
  const well = state.detail.well;
  const nextWell = nextWellAfter(well);
  state.busy = true;
  renderWellVerdict(decision);
  setMessage(`正在保存 ${well} 的人工判定…`);
  try {
    await api("/api/screening-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ well, decision, reviewer: "local_user" })
    });
    state.detail.screening = state.detail.screening || {};
    state.detail.screening.review_decision = decision;
    let updatedWell = state.wells.find(row => String(row.well).toUpperCase() === well) || null;
    for (const collection of [state.wells, state.screeningWells]) {
      const item = collection.find(row => String(row.well).toUpperCase() === well);
      if (item) {
        item.review_decision = decision;
        item.manual_review_decision = decision;
      }
    }
    renderWellVerdict(decision);
    const leavesCurrentFilter = entryFilters.manualVerdict !== "all"
      && updatedWell && !passesEntryFilters(updatedWell);
    if (pendingPlateQueue.active || leavesCurrentFilter) {
      if (pendingPlateQueue.active) state.pendingQueueVisited.add(well);
      state.busy = false;
      await loadWells(nextWell);
      setMessage(`${well} 已标记为${manualVerdictNames[decision]}，已进入下一个孔`);
    } else {
      renderWellList();
      setMessage(`${well} 已标记为${manualVerdictNames[decision]}`);
    }
  } catch (error) {
    setMessage(`人工判定保存失败：${error.message}`, true);
  } finally {
    state.busy = false;
    renderWellVerdict();
  }
}

function applyLocalScreeningUpdate(row) {
  if (!row?.well) return;
  const normalizedWell = String(row.well).toUpperCase();
  for (const item of state.screeningWells) {
    if (String(item.well).toUpperCase() === normalizedWell) Object.assign(item, row);
  }
  for (const item of state.wells) {
    if (String(item.well).toUpperCase() === normalizedWell) Object.assign(item, row);
  }
  renderWellList();
  renderPlateDialog();
}

function updateLateGrowthCard(timepoint) {
  const imageInfo = state.detail?.images?.[timepoint];
  const card = document.querySelector(`.timepoint-card[data-timepoint="${timepoint}"]`);
  if (!imageInfo || !card) return;
  card.querySelector(".timepoint-count").textContent = growthDecisionNames[imageInfo.late_growth_decision] || "待确认";
  card.querySelector(".timepoint-breakdown").textContent = lateEvidenceText(imageInfo);
  card.querySelectorAll(".late-growth-actions button").forEach(button => {
    button.classList.toggle("active", button.dataset.growth === imageInfo.late_growth_decision);
  });
}

async function loadScreeningWells() {
  state.screeningWells = await api("/api/screening-wells");
  renderPlateDialog();
}

function renderPlateDialog() {
  const grid = $("plateDialogGrid");
  if (!grid) return;
  grid.innerHTML = "<div></div>" + Array.from(
    { length: 12 }, (_, index) => `<div class="plate-axis">${index + 1}</div>`
  ).join("");
  for (const row of "ABCDEFGH") {
    grid.insertAdjacentHTML("beforeend", `<div class="plate-axis">${row}</div>`);
    for (let column = 1; column <= 12; column += 1) {
      const well = `${row}${column}`;
      const result = state.screeningWells.find(item => item.well === well);
      if (!result) {
        grid.insertAdjacentHTML("beforeend", "<div></div>");
        continue;
      }
      const button = document.createElement("button");
      button.type = "button";
      button.className = `plate-well ${result.screening_status}`;
      button.classList.toggle("current", state.detail?.well === well);
      button.textContent = well;
      button.title = [
        wellTypeNames[result.screening_status]
          || result.report_category_label
          || "T0缺失但后期出现细胞",
        result.report_reason_label || ""
      ].filter(Boolean).join("：");
      button.onclick = () => jumpFromPlate(well);
      grid.appendChild(button);
    }
  }
}

async function jumpFromPlate(well) {
  $("plateDialog").close();
  state.mode = "all";
  $("wellSearch").value = "";
  $("wellTypeFilter").value = "all";
  document.querySelectorAll("[data-mode]").forEach(node => {
    node.classList.toggle("active", node.dataset.mode === "all");
  });
  await loadWells(well);
}

function drawAll() {
  for (const timepoint of timepoints) drawTimepoint(timepoint);
}

function resetWell() {
  if (!state.detail) return;
  const removedIds = new Set(
    state.detail.objects
      .filter(object => object.is_new)
      .map(object => object.candidate_id)
  );
  state.detail.objects = state.detail.objects.filter(object => !object.is_new);
  for (const entry of state.canvases.values()) {
    entry.objects = entry.objects.filter(object => !object.is_new);
  }
  for (const object of state.detail.objects) {
    object.current_label = object.reviewed_label || object.integrated_label;
  }
  if (removedIds.has(state.selectedId)) state.selectedId = null;
  state.dirty.clear();
  state.v3TrackLabels.clear();
  drawAll();
  renderSelection();
  renderTimepointHeaders();
  $("currentWellDetail").textContent =
    `已保存 ${state.detail.objects.filter(item => item.reviewed_label).length}`
    + `/${state.detail.objects.length}；本次修改 0`;
  setMessage(`${state.detail.well} 的未保存修改已撤销`);
}

async function saveWell(approvePredictions = false) {
  if (!state.detail || state.busy) return;
  state.busy = true;
  renderUndoButton();
  const well = state.detail.well;
  const nextWell = nextWellAfter(well);
  const previousReportLabel = state.detail.report?.final_category_label || "";
  if (approvePredictions) {
    for (const object of state.detail.objects) {
      if (!object.is_new && !object.is_manual_missed) {
        object.current_label = object.integrated_label;
      }
    }
    for (const track of state.detail.v3_tracks || []) {
      if (track.label_mode !== "unified_track" || !track.conclusion) continue;
      const representative = state.detail.objects.find(
        object => String(object.v3_track_id || "") === String(track.track_id)
      );
      if (representative) applyV3TrackLabel(representative, track.conclusion);
    }
  }
  setMessage(`正在保存 ${well} 的 ${state.detail.objects.length} 个目标…`);
  try {
    await flushPendingCellCountSaves(well);
    const result = await api("/api/quick-review-well-labels", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        round_id: state.detail.round_id,
        well,
        reviewer: "local_user",
        duration_ms: state.wellLoadedAt === null ? null : Math.round(performance.now() - state.wellLoadedAt),
        items: state.detail.objects.map(object => ({
          candidate_id: object.candidate_id,
          predicted_label: object.integrated_label,
          reviewed_label: editableDecisionLabel(object),
          well: object.well,
          timepoint: object.timepoint,
          x_px: object.x_px,
          y_px: object.y_px,
          diameter_px: Number(object.diameter_px || 12),
          is_new: Boolean(object.is_new)
        })),
        v3_track_reviews: [...state.v3TrackLabels.entries()].map(([track_id, label]) => {
          const track = (state.detail.v3_tracks || []).find(
            item => String(item.track_id) === String(track_id)
          );
          return {
            track_id,
            label,
            well,
            behavior: track?.behavior || ""
          };
        })
      })
    });
    state.undoAction = result.undo_action || null;
    if (pendingPlateQueue.active) state.pendingQueueVisited.add(well);
    state.busy = false;
    renderUndoButton();
    await Promise.all([
      refreshStats(),
      loadScreeningWells(),
      loadWells(nextWell)
    ]);
    const nextReportLabel = result.report?.final_category_label || "";
    const reportChange = previousReportLabel && nextReportLabel
      && previousReportLabel !== nextReportLabel
      ? `；报告结论已由“${previousReportLabel}”更新为“${nextReportLabel}”`
      : (nextReportLabel ? `；报告结论：${nextReportLabel}` : "");
    setMessage(`${well} 已保存${reportChange}，已进入下一个孔`);
  } catch (error) {
    state.busy = false;
    renderUndoButton();
    setMessage(`保存失败：${error.message}`, true);
  }
}

async function undoLastSave() {
  if (state.busy) return;
  if (!state.undoAction) {
    await refreshUndoAction();
    if (!state.undoAction) {
      setMessage("暂无可撤销的已保存结果");
      return;
    }
  }
  if (state.dirty.size) {
    setMessage("当前孔有未保存修改，请先撤销本孔修改或保存后再执行快捷撤销", true);
    return;
  }
  const action = state.undoAction;
  state.busy = true;
  renderUndoButton();
  setMessage(`正在撤销 ${action.well} 的上一块板结果…`);
  try {
    const result = await api("/api/quick-review-undo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action_id: action.action_id, reviewer: "local_user" })
    });
    state.busy = false;
    state.undoAction = null;
    renderUndoButton();
    state.mode = "all";
    $("wellSearch").value = "";
    $("wellTypeFilter").value = "all";
    document.querySelectorAll("[data-mode]").forEach(node => {
      node.classList.toggle("active", node.dataset.mode === "all");
    });
    await Promise.all([
      refreshStats(),
      loadScreeningWells(),
      refreshUndoAction()
    ]);
    await loadWells(result.undone_action?.well || action.well);
    setMessage(`${result.undone_action?.well || action.well} 的上一块板结果已撤销`);
  } catch (error) {
    state.busy = false;
    renderUndoButton();
    setMessage(`撤销失败：${error.message}`, true);
  }
}

document.querySelectorAll("[data-mode]").forEach(button => {
  button.onclick = async () => {
    if (state.busy) return;
    state.mode = button.dataset.mode;
    document.querySelectorAll("[data-mode]").forEach(node => {
      node.classList.toggle("active", node === button);
    });
    await loadWells();
  };
});
document.querySelectorAll(".class-buttons button").forEach(button => {
  button.onclick = () => setSelectedLabel(button.dataset.label);
});
document.querySelectorAll("[data-well-verdict]").forEach(button => {
  button.onclick = () => saveManualVerdict(button.dataset.wellVerdict);
});
$("wellSearch").oninput = () => loadWells(state.detail?.well || null);
$("wellTypeFilter").onchange = () => loadWells(state.detail?.well || null);
window.addEventListener("pagehide", () => {
  for (const key of state.cellCountSaveJobs.keys()) {
    flushTimepointCellTotalSave(key);
  }
});
$("applyPlateFilter").onclick = applyPlateFilterFromEditor;
$("clearPlateFilter").onclick = () => {
  Object.values(plateFilterFields).forEach(id => { $(id).value = ""; });
  $("plateManualVerdictFilter").value = pendingPlateQueue.active ? "pending" : "all";
  applyPlateFilterFromEditor();
};
$("markerScale").oninput = drawAll;
$("diameterDown").onclick = () => changeSelectedDiameter(-2);
$("diameterUp").onclick = () => changeSelectedDiameter(2);
$("resetButton").onclick = resetWell;
$("approveButton").onclick = () => saveWell(true);
$("saveButton").onclick = () => saveWell(false);
$("undoButton").onclick = undoLastSave;
$("plateReportButton").onclick = async () => {
  if (!state.screeningWells.length) await loadScreeningWells();
  renderPlateDialog();
  $("plateDialog").showModal();
};
window.addEventListener("resize", () => {
  for (const timepoint of timepoints) fitAndDraw(timepoint);
});
document.addEventListener("keydown", event => {
  if (event.target.matches("input,select,textarea")) return;
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    undoLastSave();
    return;
  }
  if (event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveWell(false);
    return;
  }
  const labels = {
    "1": "single",
    "2": "touching_doublet",
    "3": "cluster_3plus",
    "4": "debris",
    "5": "uncertain",
    "6": "invalid"
  };
  const verdicts = {
    q: "approved",
    w: "pending",
    e: "rejected"
  };
  const verdict = verdicts[event.key.toLowerCase()];
  if (labels[event.key]) {
    event.preventDefault();
    setSelectedLabel(labels[event.key]);
  } else if (verdict) {
    event.preventDefault();
    saveManualVerdict(verdict);
  } else if (event.key === "Enter") {
    event.preventDefault();
    saveWell(false);
  }
});

const initialWell = entryFilterParameters.get("well");
const initialMode = entryFilterParameters.get("mode");
if (["pending", "reviewed", "all"].includes(initialMode)) state.mode = initialMode;
if (initialWell) state.mode = "all";
if (initialWell || initialMode) {
  document.querySelectorAll("[data-mode]").forEach(node => {
    node.classList.toggle("active", node.dataset.mode === state.mode);
  });
}
async function bootReview() {
  // The first stats call materializes the shared server-side review frame;
  // the well list and detail then reuse it instead of racing duplicate work.
  await Promise.all([refreshStats(), loadScreeningWells(), refreshUndoAction()]);
  await loadWells(initialWell?.toUpperCase() || null);
}
initializePlateFilterEditor();
initializePendingQueue();
bootReview();
