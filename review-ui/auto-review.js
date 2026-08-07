const $ = id => document.getElementById(id);
// The review UI is also mounted below /plates/<slug>.  Keep API and image
// requests inside that mounted application instead of escaping to the hub.
const mountedAppBase = (window.location.pathname.match(/^\/plates\/[^/]+/) || [""])[0];
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
  single_growth_unconfirmed: "生长待确认",
  multi_origin: "多细胞来源",
  missing_t0_or_late_object: "T0缺失/后期出现",
  no_cell_growth: "后期无明显生长",
  no_cell: "未检出细胞",
  ambiguous: "判定不明确",
  positive_control: "阳性对照"
};
const growthDecisionNames = {
  obvious_growth: "明显生长",
  no_growth: "无明显生长",
  uncertain: "无法判断",
  pending: "待确认"
};
const labelNames = {
  single: "单细胞",
  touching_doublet: "粘连2细胞",
  cluster_3plus: "3+细胞团",
  debris: "杂质/碎片",
  uncertain: "待定",
  invalid: "无关/误检"
};
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
  busy: false,
  canvases: new Map(),
  views: new Map(),
  wellLoadedAt: null,
  prefetchedDetails: new Map(),
  prefetchingDetails: new Map()
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

function pct(value) {
  return `${(Number(value || 0) * 100).toFixed(0)}%`;
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
  const loaded = await api(
    `/api/quick-review-wells?mode=${state.mode}&search=${encodeURIComponent(search)}`
  );
  const typeFilter = $("wellTypeFilter").value;
  state.wells = typeFilter === "all"
    ? loaded
    : loaded.filter(item => item.screening_status === typeFilter);
  renderWellList();
  if (!state.wells.length) {
    state.detail = null;
    state.selectedId = null;
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
        ${item.uncertain_count ? ` · ${item.uncertain_count}待定` : ""}
        ${item.temporal_review_count ? ` · ${item.temporal_review_count}时序复核` : ""}
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
  $("wellTitle").textContent = "没有待审核孔";
  $("timepointGrid").innerHTML = "";
  $("lateTimepointGrid").innerHTML = "";
  $("lateTimepointSection").hidden = true;
  $("currentWellState").textContent = "本组已完成";
  $("currentWellDetail").textContent = "可切换到“已完成”或“全部”查看";
  $("selectionEditor").classList.add("empty-selection");
  $("selectedTitle").textContent = "当前列表没有孔";
  $("selectedDetail").textContent = "";
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
  }
}

function renderWell() {
  const detail = state.detail;
  if (!detail) return;
  $("wellTitle").textContent = detail.well;
  const reviewed = detail.objects.filter(object => object.reviewed_label).length;
  const reportLabel = detail.report?.final_category_label;
  const reportReason = detail.report?.undetermined_reason_label;
  $("currentWellState").textContent =
    reportLabel || (reviewed === detail.objects.length ? "本孔已完成" : `${detail.objects.length} 个目标`);
  $("currentWellDetail").textContent =
    `已保存 ${reviewed}/${detail.objects.length}；本次修改 ${state.dirty.size}`
    + (reportReason ? `；${reportReason}` : "");
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
  renderSelection();
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
  card.classList.toggle("has-growth-overlay", Boolean(imageInfo?.growth_regions?.length));
  card.querySelector(".timepoint-name").textContent = imageInfo?.display_label || timepoint;
  card.querySelector(".timepoint-count").textContent = annotatable
    ? `${objects.length} 个目标`
    : (growthDecisionNames[imageInfo?.late_growth_decision] || "待确认");
  card.querySelector(".timepoint-breakdown").textContent = annotatable
    ? breakdownText(objects)
    : lateEvidenceText(imageInfo);
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
      applyRepresentativeLateView(timepoint);
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

function applyRepresentativeLateView(timepoint) {
  if (!lateTimepoints.includes(timepoint)) return;
  const entry = state.canvases.get(timepoint);
  const view = state.views.get(timepoint);
  if (!entry || !view || entry.defaultViewApplied) return;
  entry.defaultViewApplied = true;
  const representative = entry.imageInfo.representative_view;
  const zoom = Number(entry.imageInfo.default_zoom || 1);
  if (!representative || zoom <= 1) return;
  const baseX = Number(representative.x) / entry.imageInfo.width_px * entry.image.clientWidth;
  const baseY = Number(representative.y) / entry.imageInfo.height_px * entry.image.clientHeight;
  view.zoom = Math.min(10, zoom);
  view.panX = entry.canvas.clientWidth / 2 - baseX * view.zoom;
  view.panY = entry.canvas.clientHeight / 2 - baseY * view.zoom;
  clampView(timepoint);
  ensureHighResolution(timepoint);
  applyImageTransform(timepoint);
  drawTimepoint(timepoint);
}

function breakdownText(objects) {
  const counts = {};
  for (const object of objects) {
    counts[object.current_label] = (counts[object.current_label] || 0) + 1;
  }
  const cell = (counts.single || 0)
    + (counts.touching_doublet || 0)
    + (counts.cluster_3plus || 0);
  const parts = [`细胞 ${cell}`, `杂质 ${counts.debris || 0}`];
  if (counts.uncertain) parts.push(`待定 ${counts.uncertain}`);
  if (counts.invalid) parts.push(`排除 ${counts.invalid}`);
  return parts.join(" · ");
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
  drawGrowthRegions(ctx, entry, timepoint);
  for (const object of objects) {
    drawObject(ctx, entry, object);
  }
}

function drawGrowthRegions(ctx, entry, timepoint) {
  const regions = entry.imageInfo?.growth_regions || [];
  if (!regions.length) return;
  const view = state.views.get(timepoint) || { zoom: 1, panX: 0, panY: 0 };
  const width = entry.image.clientWidth;
  const height = entry.image.clientHeight;
  ctx.save();
  ctx.fillStyle = "rgba(37, 169, 140, .18)";
  ctx.strokeStyle = "rgba(33, 224, 179, .92)";
  ctx.lineWidth = 2;
  for (const region of regions) {
    const points = region.points || [];
    if (points.length < 3) continue;
    ctx.beginPath();
    points.forEach(([rawX, rawY], index) => {
      const x = Number(rawX) / entry.imageInfo.width_px * width * view.zoom + view.panX;
      const y = Number(rawY) / entry.imageInfo.height_px * height * view.zoom + view.panY;
      if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
  }
  ctx.restore();
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
  const label = object.current_label;
  const suspectedDead = Boolean(object.v2_suspected_dead_cell)
    && ["single", "touching_doublet", "cluster_3plus"].includes(label);
  const color = suspectedDead ? "#ef476f" : (labelColors[label] || labelColors.uncertain);
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
  ctx.setLineDash(label === "uncertain" ? [5, 4] : (suspectedDead ? [3, 2] : []));
  const contour = contourGeometry(entry, object);
  if (contour.length >= 3 && label !== "invalid") {
    ctx.beginPath();
    const firstMid = [
      (contour[0][0] + contour[1][0]) / 2,
      (contour[0][1] + contour[1][1]) / 2
    ];
    ctx.moveTo(firstMid[0], firstMid[1]);
    for (let index = 1; index <= contour.length; index += 1) {
      const current = contour[index % contour.length];
      const next = contour[(index + 1) % contour.length];
      const midpoint = [(current[0] + next[0]) / 2, (current[1] + next[1]) / 2];
      ctx.quadraticCurveTo(current[0], current[1], midpoint[0], midpoint[1]);
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
  const names = {
    static_pixel_identity_debris: "\u8de8\u65f6\u95f4\u50cf\u7d20\u7a33\u5b9a\uff0c\u503e\u5411\u6742\u8d28",
    changing_shape_or_growth_cell: "\u8de8\u65f6\u95f4\u53d1\u751f\u53d8\u5316\uff0c\u503e\u5411\u7ec6\u80de",
    ambiguous_temporal_appearance: "\u8de8\u65f6\u95f4\u8bc1\u636e\u4ecd\u4e0d\u8db3",
    insufficient_evidence: "\u672a\u627e\u5230\u8db3\u591f\u7684\u5e73\u884c\u5bf9\u7167",
  };
  const text = names[object.temporal_appearance_status] || "";
  return text
    ? ` \u00b7 ${text} (${Number(object.temporal_appearance_match_count || 0)}\u5f20)`
    : "";
}

function renderSelection() {
  const editor = $("selectionEditor");
  const object = selectedObject();
  if (!object) {
    editor.classList.add("empty-selection");
    $("selectedTitle").textContent = "点击图中的标记进行判定";
    $("selectedDetail").textContent =
      "绿色为单细胞，青色为粘连双细胞，紫色为多细胞团，橙色为杂质，黄色虚线圆为待定目标。";
    $("selectedProbabilities").innerHTML = "";
    $("selectedPatch").removeAttribute("src");
    $("diameterControl").hidden = true;
    document.querySelectorAll(".class-buttons button").forEach(
      button => button.classList.remove("active")
    );
    return;
  }
  editor.classList.remove("empty-selection");
  $("selectedTitle").textContent =
    `${object.timepoint} · ${labelNames[object.current_label]}`
    + (object.v2_suspected_dead_cell ? " · 疑似死细胞" : "");
  $("selectedDetail").textContent =
    `${object.candidate_id} · 位置 (${Math.round(object.x_px)}, ${Math.round(object.y_px)})`
    + ` · 直径 ${Number(object.diameter_px || 0).toFixed(1)} px`
    + (
      object.temporal_completion_status
      && object.temporal_completion_status !== "not_evaluated"
        ? ` · 时序${object.temporal_completion_status === "auto_promoted" ? "自动补回" : "复核"}`
        : ""
    )
    + (
      Number(object.temporal_completion_score || 0) > 0
        ? ` ${pct(object.temporal_completion_score)}`
        : ""
    )
    + (object.temporal_completion_direction
      ? ` · ${object.temporal_completion_direction}` : "")
    + (state.dirty.has(object.candidate_id) ? " · 尚未保存" : "");
  $("selectedDetail").textContent += temporalAppearanceText(object);
  $("selectedPatch").src = appUrl(
    `/api/patch?well=${object.well}&timepoint=${object.timepoint}`
    + `&x=${object.x_px}&y=${object.y_px}&size=256`
  );
  $("selectedProbabilities").innerHTML = `
    <span>细胞 ${pct(object.cell_probability)}</span>
    <span>杂质 ${pct(object.debris_probability)}</span>
    <span>无效 ${pct(object.invalid_probability)}</span>
    <span>单细胞 ${pct(object.single_probability)}</span>
    <span>双细胞 ${pct(object.touching_doublet_probability)}</span>
    <span>3+ ${pct(object.cluster_3plus_probability)}</span>
    ${object.v2_temporal_same_object_score !== undefined ? `<span>同一对象 ${pct(object.v2_temporal_same_object_score)}</span>` : ""}
    ${object.v2_temporal_static_similarity_score !== undefined ? `<span>静态相似 ${pct(object.v2_temporal_static_similarity_score)}</span>` : ""}
    ${object.v2_temporal_candidate_count !== undefined ? `<span>已匹配时点 ${Number(object.v2_temporal_candidate_count || 0)}/3</span>` : ""}
    ${object.v2_suspected_dead_cell ? `<span>疑似死细胞 ${pct(object.v2_suspected_dead_cell_score)}</span>` : ""}
    ${object.v2_temporal_debris_boost > 0 ? `<span>时序向杂质修正 +${pct(object.v2_temporal_debris_boost)}</span>` : ""}
    ${object.v2_adjusted_cell_probability !== undefined ? `<span>调整后细胞 ${pct(object.v2_adjusted_cell_probability)}</span>` : ""}
    ${object.v2_adjusted_debris_probability !== undefined ? `<span>调整后杂质 ${pct(object.v2_adjusted_debris_probability)}</span>` : ""}
    ${object.v2_temporal_recovered ? `<span>时序恢复候选</span>` : ""}
    ${object.v2_temporal_reason ? `<span>时序依据：${({applied: "已应用相似性修正", temporal_recovery: "由其他时点恢复", noncell_debris_resolution: "低细胞概率，按概率判为杂质", noncell_invalid_resolution: "低细胞概率，按概率判为无效", unary_debris_probability_dominant: "单帧杂质概率高于细胞，按杂质判定", high_confidence_base: "单帧高置信，未介入", high_confidence_cell_anchor: "匹配到高置信细胞锚点，向细胞修正", high_confidence_debris_anchor: "匹配到高置信杂质锚点，向杂质修正", division_or_growth_cell_evidence: "检测到时序修正前的细胞分裂或生长证据", multi_frame_cell_consensus: "多帧形态共同支持细胞", multi_frame_debris_consensus: "多帧形态共同支持杂质", multi_frame_debris_probability_trend: "三帧杂质概率持续升高，向杂质修正", two_frame_static_object: "两帧物体稳定，提供弱杂质证据", three_frame_static_object: "三帧物体稳定，提供强杂质证据", three_frame_static_debris_consensus: "三帧同一物体高度稳定，跨越单帧阈值向杂质修正", suspected_dead_cell: "三帧细胞形态高度稳定，标记为疑似死细胞", suspected_dead_revoked_by_later_division: "T0唯一细胞后续出现可信分裂，已撤销疑似死细胞", conflicting_temporal_evidence: "时序证据冲突，维持原判定", outside_temporal_ambiguity_band: "三帧虽已匹配，但不足以覆盖明确的单帧结论", no_decisive_object_evidence: "三帧虽已匹配，但分类证据未形成一致结论", object_change_weak_cell_evidence: "物体发生变化，提供弱细胞证据", low_match: "跨时点匹配不足", low_similarity: "静态相似不足", invalid_candidate: "已由无效模型处理", insufficient_parallel_candidates: "近似位置的匹配时点不足，维持原判定", insufficient_frames: "可用时点不足", static_wall_artifact: "静态孔壁伪目标", not_evaluated: "未评估"})[object.v2_temporal_reason] || object.v2_temporal_reason}</span>` : ""}
  `;
  const canResize = Boolean(object.is_new || object.is_manual_missed);
  $("diameterControl").hidden = !canResize;
  $("diameterValue").textContent =
    `${Number(object.diameter_px || 12).toFixed(0)} px`;
  document.querySelectorAll(".class-buttons button").forEach(button => {
    button.classList.toggle("active", button.dataset.label === object.current_label);
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
  object.current_label = label;
  if (
    label === object.integrated_label
    && !object.reviewed_label
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
  });
}

async function saveLateGrowthDecision(timepoint, decision) {
  if (!state.detail || state.busy) return;
  const well = state.detail.well;
  state.busy = true;
  setMessage(`正在保存 ${well} ${timepoint} 生长判定…`);
  try {
    await api("/api/late-growth-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        well, timepoint, decision, reviewer: "local_user"
      })
    });
    state.busy = false;
    await loadScreeningWells();
    await loadWells(well);
    setMessage(`${well} ${timepoint} 已标记为${growthDecisionNames[decision]}`);
  } catch (error) {
    state.busy = false;
    setMessage(`生长判定保存失败：${error.message}`, true);
  }
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
        result.report_category_label || wellTypeNames[result.screening_status] || "判定不明确",
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
  const well = state.detail.well;
  const nextWell = nextWellAfter(well);
  const previousReportLabel = state.detail.report?.final_category_label || "";
  if (approvePredictions) {
    for (const object of state.detail.objects) {
      if (!object.is_new && !object.is_manual_missed) {
        object.current_label = object.integrated_label;
      }
    }
  }
  setMessage(`正在保存 ${well} 的 ${state.detail.objects.length} 个目标…`);
  try {
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
          reviewed_label: object.current_label,
          well: object.well,
          timepoint: object.timepoint,
          x_px: object.x_px,
          y_px: object.y_px,
          diameter_px: Number(object.diameter_px || 12),
          is_new: Boolean(object.is_new)
        }))
      })
    });
    state.busy = false;
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
    setMessage(`保存失败：${error.message}`, true);
  }
}

async function generateNextRound() {
  if (state.busy) return;
  state.busy = true;
  $("newRoundButton").disabled = true;
  setMessage("正在用已审核结果训练形态与粘连模型，并更新下一轮孔级判定…");
  try {
    const result = await api("/api/integrated-review-new-round", {
      method: "POST"
    });
    setMessage(`下一轮已生成：${result.integrated_round.round_id}`);
    state.mode = "pending";
    document.querySelectorAll("[data-mode]").forEach(button => {
      button.classList.toggle("active", button.dataset.mode === state.mode);
    });
    state.busy = false;
    await refreshStats();
    await loadWells();
  } catch (error) {
    state.busy = false;
    setMessage(`训练失败：${error.message}`, true);
  } finally {
    $("newRoundButton").disabled = false;
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
$("wellSearch").oninput = () => loadWells(state.detail?.well || null);
$("wellTypeFilter").onchange = () => loadWells(state.detail?.well || null);
$("markerScale").oninput = drawAll;
$("diameterDown").onclick = () => changeSelectedDiameter(-2);
$("diameterUp").onclick = () => changeSelectedDiameter(2);
$("resetButton").onclick = resetWell;
$("approveButton").onclick = () => saveWell(true);
$("saveButton").onclick = () => saveWell(false);
$("newRoundButton").onclick = generateNextRound;
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
  const labels = {
    "1": "single",
    "2": "touching_doublet",
    "3": "cluster_3plus",
    "4": "debris",
    "5": "uncertain",
    "0": "invalid"
  };
  if (labels[event.key]) {
    event.preventDefault();
    setSelectedLabel(labels[event.key]);
  } else if (event.key === "Enter") {
    event.preventDefault();
    saveWell(false);
  }
});

const initialWell = new URLSearchParams(window.location.search).get("well");
if (initialWell) {
  state.mode = "all";
  document.querySelectorAll("[data-mode]").forEach(node => {
    node.classList.toggle("active", node.dataset.mode === "all");
  });
}
async function bootReview() {
  // The first stats call materializes the shared server-side review frame;
  // the well list and detail then reuse it instead of racing duplicate work.
  await Promise.all([refreshStats(), loadScreeningWells()]);
  await loadWells(initialWell?.toUpperCase() || null);
}
bootReview();
