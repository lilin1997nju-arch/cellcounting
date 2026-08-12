const $ = (id) => document.getElementById(id);

const scope = document.querySelector('meta[name="review-scope"]')?.content || "plate";
const projectName = document.querySelector('meta[name="project-name"]')?.content || "当前板子";
const pagePath = window.location.pathname.replace(/\/+$/, "");
const plateBase = scope === "project" ? "" : pagePath.replace(/\/single-doublet-review$/, "");
const apiRoot = scope === "project" ? "" : plateBase;
const endpoints = {
  queue: scope === "project" ? "/api/multiplicity-training-candidates" : `${apiRoot}/api/multiplicity-candidates`,
  stats: scope === "project" ? "/api/multiplicity-training-stats" : `${apiRoot}/api/multiplicity-stats`,
  labels: scope === "project" ? "/api/multiplicity-training-labels" : `${apiRoot}/api/multiplicity-labels`,
};

const categories = [
  { key: "single", title: "单细胞", description: "检查模型认为只有一个细胞中心的候选，重点挑出实际是粘连或杂质的样本。", color: "single-dot" },
  { key: "touching_doublet", title: "粘连双细胞", description: "重点看两个细胞中心、双叶轮廓或中间凹陷；把真正的单细胞和 3+ 团挑出来纠正。", color: "doublet-dot" },
  { key: "cluster_3plus", title: "3+细胞团", description: "检查多中心或较大团块，必要时纠正为粘连双细胞、单细胞或非细胞。", color: "cluster-dot" },
  { key: "debris", title: "杂质", description: "检查模型判定为杂质的区域，注意不要把仍有清晰细胞轮廓的样本留在这里。", color: "debris-dot" },
  { key: "invalid", title: "无效（孔壁）", description: "这里只呈现模型判定为孔壁无效的候选，用于挑出被孔壁误杀的真实细胞。", color: "invalid-dot" },
];
const categoryByKey = Object.fromEntries(categories.map((item) => [item.key, item]));
const state = {
  activeCategory: "single",
  pageSize: 24,
  batches: Object.fromEntries(categories.map((item) => [item.key, []])),
  stats: null,
  busy: false,
};

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = await response.text();
    try { detail = JSON.parse(detail).detail || detail; } catch (_) { /* plain text */ }
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return response.json();
}

function setToast(message, error = false) {
  const toast = $("toast");
  toast.textContent = message;
  toast.style.background = error ? "#7b403f" : "#0c3f47";
  toast.classList.add("show");
  window.clearTimeout(setToast.timer);
  setToast.timer = window.setTimeout(() => toast.classList.remove("show"), 3000);
}

function number(value, fallback = null) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function probability(value) {
  const parsed = number(value);
  return parsed === null ? null : Math.max(0, Math.min(1, parsed));
}

function pct(value) {
  const parsed = probability(value);
  return parsed === null ? "—" : `${Math.round(parsed * 100)}%`;
}

function formatNumber(value, digits = 0) {
  const parsed = number(value);
  return parsed === null ? "—" : parsed.toLocaleString("zh-CN", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function patchUrl(item) {
  const base = scope === "project" ? (item.patch_base || "") : apiRoot;
  const params = new URLSearchParams({
    well: String(item.well || ""),
    timepoint: String(item.timepoint || ""),
    x: String(item.x_px ?? 0),
    y: String(item.y_px ?? 0),
    size: "320",
  });
  return `${base}/api/patch?${params.toString()}`;
}

function categoryConfidence(item, category) {
  const column = {
    single: "single_probability",
    touching_doublet: "touching_doublet_probability",
    cluster_3plus: "cluster_3plus_probability",
    debris: "debris_probability",
    invalid: "invalid_probability",
  }[category];
  return probability(item[column]);
}

function parseContour(item) {
  let points;
  try {
    points = Array.isArray(item.v2_contour_json)
      ? item.v2_contour_json
      : JSON.parse(item.v2_contour_json || "[]");
  } catch (_) {
    return [];
  }
  if (!Array.isArray(points) || points.length < 3) return [];
  const size = 320;
  const centerX = number(item.x_px, 0);
  const centerY = number(item.y_px, 0);
  const coords = points
    .map((point) => [number(point?.[0]), number(point?.[1])])
    .filter((point) => point.every((value) => value !== null));
  if (coords.length < 3) return [];

  // Current V2 artifacts store image-space coordinates.  A few early test
  // artifacts stored local mask coordinates; detect those and center them in
  // the crop instead of drawing them in the top-left corner.
  const meanX = coords.reduce((sum, point) => sum + point[0], 0) / coords.length;
  const meanY = coords.reduce((sum, point) => sum + point[1], 0) / coords.length;
  const imageSpace = Math.abs(meanX - centerX) < size * 0.75 && Math.abs(meanY - centerY) < size * 0.75;
  const cropLeft = centerX - size / 2;
  const cropTop = centerY - size / 2;
  return coords.map(([x, y]) => {
    const localX = imageSpace ? x - cropLeft : x + size / 2;
    const localY = imageSpace ? y - cropTop : y + size / 2;
    return `${Math.max(0, Math.min(size, localX)).toFixed(1)},${Math.max(0, Math.min(size, localY)).toFixed(1)}`;
  });
}

function renderContour(card, item) {
  const line = card.querySelector(".contour-line");
  const points = parseContour(item);
  line.setAttribute("points", points.join(" "));
  line.style.opacity = points.length >= 3 ? "0.92" : "0";
  card.querySelector(".thumb-caption").textContent = points.length >= 3 ? "V2 CONTOUR" : "CENTER MARK";
}

function renderContourVisibility(card, item) {
  const show = item.showContour !== false;
  const overlay = card.querySelector(".contour-overlay");
  const toggle = card.querySelector(".contour-toggle");
  overlay.classList.toggle("is-hidden", !show);
  toggle.textContent = show ? "隐藏轮廓" : "显示轮廓";
  toggle.setAttribute("aria-pressed", String(show));
  toggle.classList.toggle("active", show);
}

function applyDecision(card, item, label, confirmed = false) {
  if (confirmed) item.confirmed = true;
  item.correction = label || null;
  const correctButton = card.querySelector(".correct-button");
  correctButton.hidden = !item.correction;
  correctButton.classList.toggle("active", Boolean(item.confirmed) && !item.correction);
  card.querySelectorAll("[data-correct-label]").forEach((button) => {
    button.classList.toggle("active", button.dataset.correctLabel === item.correction);
  });
  const stateLabel = card.querySelector(".decision-state");
  stateLabel.textContent = item.correction
    ? `待提交纠正：${categoryByKey[item.correction]?.title || item.correction}`
    : item.confirmed ? "默认正确（待提交）" : "未选择纠正";
  stateLabel.classList.toggle("correction", Boolean(item.correction));
  renderPendingCount();
}

function renderCard(item, category) {
  const card = document.getElementById("cardTemplate").content.firstElementChild.cloneNode(true);
  card.dataset.reviewId = item.review_id || `${item.plate_slug || "plate"}::${item.candidate_id}`;
  card.querySelector(".patch-image").src = patchUrl(item);
  card.querySelector(".patch-image").alt = `${item.well || ""} ${item.timepoint || ""} 候选图像`;
  renderContour(card, item);
  card.querySelector(".plate-label").textContent = item.plate_label || projectName;
  card.querySelector(".well-label").textContent = `${item.well || "—"} · ${item.timepoint || "—"}`;
  card.querySelector(".model-badge").textContent = `模型：${categoryByKey[category]?.title || category}`;
  card.querySelector(".candidate-id").textContent = item.candidate_id || "—";
  card.querySelector(".confidence-value").textContent = pct(categoryConfidence(item, category));
  card.querySelectorAll("[data-prob]").forEach((node) => {
    node.textContent = pct(item[node.dataset.prob]);
  });
  card.querySelector(".area-value").textContent = formatNumber(item.area_px);
  card.querySelector(".correct-button").addEventListener("click", () => applyDecision(card, item, null, true));
  card.querySelector(".contour-toggle").addEventListener("click", () => {
    item.showContour = item.showContour === false;
    renderContourVisibility(card, item);
  });
  card.querySelectorAll("[data-correct-label]").forEach((button) => {
    button.addEventListener("click", () => applyDecision(card, item, button.dataset.correctLabel, true));
  });
  // A card with no correction is implicitly confirmed.  The reviewer only
  // needs to click when the model's category is wrong.
  applyDecision(card, item, item.correction, true);
  renderContourVisibility(card, item);
  return card;
}

function activeBatchItems() {
  return state.batches[state.activeCategory] || [];
}

function pendingItems() {
  return activeBatchItems().filter((item) => item.correction);
}

function renderPendingCount() {
  // Submission advances the whole visible batch.  The badge remains the
  // number of explicit corrections so the reviewer can see how many cards
  // differ from the model's default category.
  const count = pendingItems().length;
  $("pendingCount").textContent = String(count);
  $("submitButton").disabled = state.busy || activeBatchItems().length === 0;
}

function renderActiveBatch() {
  const category = categoryByKey[state.activeCategory];
  const items = state.batches[state.activeCategory] || [];
  $("categoryTitle").textContent = category.title;
  $("categoryDescription").textContent = category.description;
  $("visibleCount").textContent = String(items.length);
  $("emptyState").hidden = items.length !== 0;
  const grid = $("cardGrid");
  grid.replaceChildren(...items.map((item) => renderCard(item, state.activeCategory)));
  document.querySelectorAll(".category-tab").forEach((button) => {
    const active = button.dataset.category === state.activeCategory;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  });
  renderPendingCount();
}

function renderTabCounts() {
  categories.forEach(({ key }) => {
    const target = document.querySelector(`[data-count="${key}"]`);
    if (target) target.textContent = String((state.batches[key] || []).length);
  });
}

function renderStats(stats) {
  state.stats = stats || { counts: {} };
  const counts = state.stats.counts || {};
  const values = {
    savedSingle: counts.single,
    savedDoublet: counts.touching_doublet,
    savedCluster: counts.cluster_3plus,
    savedDebris: counts.debris,
    savedInvalid: counts.invalid,
  };
  Object.entries(values).forEach(([id, value]) => { $(id).textContent = formatNumber(value || 0); });
  const single = Number(counts.single || 0);
  const doublet = Number(counts.touching_doublet || 0);
  const quota = Math.min(100, (single / 40 + doublet / 20) / 2 * 100);
  $("quotaBar").style.width = `${quota}%`;
  $("quotaDetail").textContent = `当前：${single}/40 个清晰单细胞 + ${doublet}/20 个清晰粘连双细胞`;
  if (single >= 40 && doublet >= 20) {
    $("quotaCopy").textContent = "单双细胞定向配额已达到最低目标；仍可继续收集边界样本。";
  }
}

async function loadStats() {
  try {
    renderStats(await api(endpoints.stats));
  } catch (error) {
    setToast(`统计加载失败：${error.message}`, true);
  }
}

async function loadBatches(force = false) {
  if (state.busy && !force) return;
  state.busy = true;
  renderPendingCount();
  $("queueStatus").textContent = "正在读取五类候选…";
  try {
    const size = Math.max(1, Math.min(48, Number($("pageSize").value || state.pageSize)));
    state.pageSize = size;
    const results = await Promise.all(categories.map(async ({ key }) => {
      const query = new URLSearchParams({ mode: "uncertain", limit: String(size), category: key });
      const rows = await api(`${endpoints.queue}?${query.toString()}`);
      if (rows.length && !Object.prototype.hasOwnProperty.call(rows[0], "predicted_category")) {
        throw new Error("服务端尚未加载分类审核接口，请重启 8777 后刷新页面");
      }
      return [key, rows];
    }));
    results.forEach(([key, items]) => { state.batches[key] = Array.isArray(items) ? items : []; });
    renderTabCounts();
    renderActiveBatch();
    const total = categories.reduce((sum, { key }) => sum + state.batches[key].length, 0);
    $("queueStatus").textContent = `已加载 ${total} 个候选，当前显示“${categoryByKey[state.activeCategory].title}”`;
  } catch (error) {
    $("queueStatus").textContent = "候选加载失败";
    setToast(`候选加载失败：${error.message}`, true);
  } finally {
    state.busy = false;
    renderPendingCount();
  }
}

async function submitCorrections() {
  const reviewed = activeBatchItems();
  if (!reviewed.length || state.busy) return;
  state.busy = true;
  renderPendingCount();
  try {
    const items = reviewed.map((item) => {
      const corrected = Boolean(item.correction);
      const value = {
        candidate_id: item.candidate_id,
        well: item.well,
        timepoint: item.timepoint,
        x_px: Number(item.x_px || 0),
        y_px: Number(item.y_px || 0),
        // The reviewer has inspected every card in the visible batch.  An
        // unchanged card therefore uses the model's category as a confirmed
        // human label; a changed card uses the selected correction instead.
        label: item.correction || item.predicted_category || state.activeCategory,
        source: corrected ? "categorized_batch_review" : "categorized_batch_confirmed",
      };
      if (scope === "project") value.plate_slug = item.plate_slug;
      return value;
    });
    const payload = { items, reviewer: "local_user" };
    const result = await api(endpoints.labels, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const correctedCount = reviewed.filter((item) => item.correction).length;
    setToast(`本批 ${result.saved || items.length} 个样本已完成；其中 ${correctedCount} 个纠正`);
    state.busy = false;
    await Promise.all([loadStats(), loadBatches(true)]);
  } catch (error) {
    setToast(`本批提交失败：${error.message}`, true);
  } finally {
    state.busy = false;
    renderPendingCount();
  }
}

document.querySelectorAll(".category-tab").forEach((button) => {
  button.addEventListener("click", () => {
    if (state.busy) return;
    state.activeCategory = button.dataset.category;
    renderActiveBatch();
    $("queueStatus").textContent = `当前显示“${categoryByKey[state.activeCategory].title}”`;
  });
});
$("pageSize").addEventListener("change", loadBatches);
$("reloadButton").addEventListener("click", loadBatches);
$("submitButton").addEventListener("click", submitCorrections);

$("scopeBadge").textContent = scope === "project" ? projectName : projectName;
$("queueScope").textContent = scope === "project" ? "项目级 · 分板子交错取样" : "当前板子";
const projectBackUrl = document.querySelector('meta[name="project-back-url"]')?.content;
if (scope === "project") {
  $("backLink").href = projectBackUrl || "/";
  $("backLink").textContent = "返回项目汇总";
  $("legacyLink").hidden = true;
}

Promise.all([loadStats(), loadBatches()]);
