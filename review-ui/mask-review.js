(() => {
  const SIZE = 96;
  const API_BASE = document.querySelector('meta[name="mask-review-base"]')?.content || "";
  const PROJECT_ID = document.querySelector('meta[name="project-id"]')?.content || "";
  const PROJECT_BACK_URL = document.querySelector('meta[name="project-back-url"]')?.content || "/";
  const backLink = document.querySelector(".back-link");
  if (backLink) backLink.href = PROJECT_BACK_URL;
  const apiPath = (path) => {
    const base = API_BASE.trim();
    const separator = path.includes("?") ? "&" : "?";
    const scopedPath = PROJECT_ID ? `${path}${separator}project_id=${encodeURIComponent(PROJECT_ID)}` : path;
    if (!base || base === "/") return scopedPath;
    return `${base.replace(/\/+$/, "")}${scopedPath}`;
  };
  const state = {
    rounds: [],
    roundId: "",
    reviewMode: "p0",
    comparisonOptions: null,
    candidates: [],
    currentIndex: -1,
    item: null,
    modelMask: new Uint8Array(SIZE * SIZE),
    oldModelMask: new Uint8Array(SIZE * SIZE),
    newModelMask: new Uint8Array(SIZE * SIZE),
    currentMask: new Uint8Array(SIZE * SIZE),
    editing: false,
    erasing: false,
    brushSize: 3,
    drawing: false,
    imageRequestId: 0,
  };

  const $ = (id) => document.getElementById(id);
  const sourceImage = $("source-image");
  const canvas = $("mask-canvas");
  const ctx = canvas.getContext("2d");

  function setMessage(message, error = false) {
    const element = $("message");
    element.textContent = message || "";
    element.classList.toggle("error", error);
  }

  function boundaryMessage() {
    return state.reviewMode === "comparison"
      ? "原图已加载。橙色为旧模型、黄色为新模型、绿色为当前审核边界。"
      : "原图已加载。黄色为模型边界，绿色为当前审核边界。";
  }

  async function api(url, options) {
    const response = await fetch(apiPath(url), options);
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `请求失败 (${response.status})`);
    return payload;
  }

  function decodeRle(value) {
    const mask = new Uint8Array(SIZE * SIZE);
    let runs = [];
    try { runs = JSON.parse(value || "[]"); } catch { return mask; }
    for (const run of runs) {
      if (!Array.isArray(run) || run.length !== 2) continue;
      const start = Math.max(0, Number(run[0]) | 0);
      const end = Math.min(mask.length, start + Math.max(0, Number(run[1]) | 0));
      for (let index = start; index < end; index += 1) mask[index] = 1;
    }
    return mask;
  }

  function encodeRle(mask) {
    const runs = [];
    let start = -1;
    for (let index = 0; index <= mask.length; index += 1) {
      const active = index < mask.length && mask[index] === 1;
      if (active && start < 0) start = index;
      if (!active && start >= 0) {
        runs.push([start, index - start]);
        start = -1;
      }
    }
    return JSON.stringify(runs);
  }

  function masksEqual(first, second) {
    if (first.length !== second.length) return false;
    for (let index = 0; index < first.length; index += 1) {
      if (first[index] !== second[index]) return false;
    }
    return true;
  }

  function boundary(mask) {
    const points = [];
    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        const index = y * SIZE + x;
        if (!mask[index]) continue;
        if (x === 0 || x === SIZE - 1 || y === 0 || y === SIZE - 1 ||
            !mask[index - 1] || !mask[index + 1] || !mask[index - SIZE] || !mask[index + SIZE]) {
          points.push([x, y]);
        }
      }
    }
    return points;
  }

  function drawBoundary(mask, color, width) {
    ctx.fillStyle = color;
    for (const [x, y] of boundary(mask)) ctx.fillRect(x, y, width, width);
  }

  function drawCanvas() {
    ctx.clearRect(0, 0, SIZE, SIZE);
    const overlay = ctx.createImageData(SIZE, SIZE);
    for (let index = 0; index < state.currentMask.length; index += 1) {
      if (!state.currentMask[index]) continue;
      overlay.data[index * 4] = 56;
      overlay.data[index * 4 + 1] = 224;
      overlay.data[index * 4 + 2] = 189;
      overlay.data[index * 4 + 3] = 88;
    }
    ctx.putImageData(overlay, 0, 0);
    if (state.reviewMode === "comparison") {
      drawBoundary(state.oldModelMask, "rgba(255, 140, 105, .98)", 1);
      drawBoundary(state.newModelMask, "rgba(255, 209, 102, .98)", 1);
    } else {
      drawBoundary(state.modelMask, "rgba(255, 209, 102, .95)", 1);
    }
    drawBoundary(state.currentMask, "rgba(66, 224, 189, .98)", 1);
  }

  function updateModeUi() {
    const comparison = state.reviewMode === "comparison";
    $("comparison-setup").hidden = !comparison;
    $("accept-old-button").hidden = !comparison;
    $("old-model-legend").hidden = !comparison;
    $("new-model-legend").querySelector(".legend-model").hidden = false;
    $("new-model-legend").lastChild.textContent = comparison ? "新模型边界" : "模型边界";
    $("accept-button").textContent = comparison ? "采用新模型轮廓" : "接受模型轮廓";
  }

  function setEditMode(enabled) {
    state.editing = Boolean(enabled);
    $("edit-button").textContent = state.editing ? "结束手动修正" : "开始手动修正";
    $("edit-state").textContent = state.editing ? (state.erasing ? "橡皮模式" : "画笔模式") : "浏览模式";
    canvas.style.cursor = state.editing ? "crosshair" : "default";
  }

  function updateDetails() {
    const item = state.item;
    if (!item) {
      $("candidate-title").textContent = "尚未选择对象";
      $("decision-badge").textContent = "待审核";
      $("candidate-details").innerHTML = "";
      $("canvas-empty").style.display = "block";
      $("accept-button").disabled = true;
      $("reject-button").disabled = true;
      $("save-button").disabled = true;
      $("accept-old-button").disabled = true;
      return;
    }
    $("canvas-empty").style.display = "none";
    $("accept-button").disabled = false;
    $("reject-button").disabled = false;
    $("save-button").disabled = false;
    $("accept-old-button").disabled = state.reviewMode !== "comparison";
    $("candidate-title").textContent = `${item.well} / ${item.timepoint} / ${item.candidate_id}`;
    $("decision-badge").textContent = item.decision === "pending" ? "待审核" : item.decision;
    const values = [
      ["模型类别", item.integrated_label],
      [state.reviewMode === "comparison" ? "新模型置信度" : "模型置信度", Number(item.model_confidence).toFixed(3)],
      [state.reviewMode === "comparison" ? "新模型面积" : "模型面积", `${item.model_area_px} px²`],
      ["当前面积", `${item.reviewed_area_px} px²`],
      ["细胞概率", Number(item.cell_probability).toFixed(3)],
      ["无效概率", Number(item.invalid_probability).toFixed(3)],
      ["P0 修正状态", item.refinement_status || "—"],
      ["面积变化比", Number(item.refinement_area_ratio).toFixed(3)],
      ["P0 IoU", Number(item.refinement_iou).toFixed(3)],
      ["坐标", `(${Number(item.x_px).toFixed(1)}, ${Number(item.y_px).toFixed(1)})`],
    ];
    if (state.reviewMode === "comparison") {
      values.splice(3, 0,
        ["旧模型面积", `${item.old_model_area_px ?? 0} px²`],
        ["旧模型置信度", Number(item.old_model_confidence ?? 0).toFixed(3)],
      );
    }
    $("candidate-details").innerHTML = values.map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join("");
    $("notes").value = item.notes || "";
  }

  function renderSummary(summary) {
    $("summary").innerHTML = [
      ["待审核", summary.pending_count || 0],
      ["已接受", summary.accepted_count || 0],
      ["已修改", summary.edited_count || 0],
      ["已拒绝", summary.rejected_count || 0],
    ].map(([label, value]) => `<div class="summary-card"><strong>${value}</strong><small>${label}</small></div>`).join("");
  }

  function renderQueue() {
    const list = $("queue-list");
    $("queue-count").textContent = String(state.candidates.length);
    $("queue-caption").textContent = state.candidates.length ? "按风险优先级排序" : "当前筛选无对象";
    list.innerHTML = "";
    state.candidates.forEach((candidate, index) => {
      const button = document.createElement("button");
      button.className = `queue-item ${index === state.currentIndex ? "selected" : ""}`;
      const comparisonMeta = state.reviewMode === "comparison"
        ? `旧 ${candidate.old_model_area_px ?? "?"} / 新 ${candidate.model_area_px ?? "?"}px²`
        : `${candidate.model_area_px}px²`;
      button.innerHTML = `<span><span class="candidate-name">${candidate.well} · ${candidate.timepoint}</span><br><span class="candidate-meta">${candidate.candidate_id} · ${candidate.integrated_label} · ${comparisonMeta}</span></span><span class="candidate-status">${candidate.decision}</span>`;
      button.addEventListener("click", () => selectCandidate(index));
      list.appendChild(button);
    });
  }

  async function loadQueue(preferCandidateId = "") {
    if (!state.roundId) return;
    const status = $("status-select").value;
    const [candidates, summary] = await Promise.all([
      api(`/api/mask-review-candidates?round_id=${encodeURIComponent(state.roundId)}&status=${encodeURIComponent(status)}&limit=5000`),
      api(`/api/mask-review-summary?round_id=${encodeURIComponent(state.roundId)}`),
    ]);
    state.candidates = candidates;
    renderSummary(summary);
    const target = preferCandidateId ? candidates.findIndex((item) => item.candidate_id === preferCandidateId) : -1;
    state.currentIndex = target >= 0 ? target : Math.min(Math.max(state.currentIndex, 0), candidates.length - 1);
    renderQueue();
    if (state.currentIndex >= 0 && candidates[state.currentIndex]) await selectCandidate(state.currentIndex, false);
    else {
      state.item = null;
      sourceImage.hidden = true;
      sourceImage.removeAttribute("src");
      updateDetails();
      drawCanvas();
    }
  }

  async function selectCandidate(index, refreshList = true) {
    if (index < 0 || index >= state.candidates.length) return;
    state.currentIndex = index;
    const candidate = state.candidates[index];
    if (refreshList) renderQueue();
    setMessage("加载对象…");
    try {
      state.item = await api(`/api/mask-review-candidate?round_id=${encodeURIComponent(state.roundId)}&candidate_id=${encodeURIComponent(candidate.candidate_id)}`);
      state.oldModelMask = decodeRle(state.item.old_model_mask_rle || state.item.model_mask_rle);
      state.newModelMask = decodeRle(state.item.new_model_mask_rle || state.item.model_mask_rle);
      state.modelMask = state.newModelMask.slice();
      state.currentMask = decodeRle(state.item.reviewed_mask_rle || state.item.model_mask_rle);
      const imageRequestId = state.imageRequestId + 1;
      state.imageRequestId = imageRequestId;
      sourceImage.hidden = true;
      sourceImage.onload = () => {
        if (state.imageRequestId !== imageRequestId) return;
        sourceImage.hidden = false;
        drawCanvas();
        setMessage(boundaryMessage());
      };
      sourceImage.onerror = () => {
        if (state.imageRequestId !== imageRequestId) return;
        sourceImage.hidden = true;
        drawCanvas();
        setMessage("原图加载失败，但仍可审核 Mask；请检查当前板的图像清单或服务状态。", true);
      };
      const sourceQuery = state.item.source_config
        ? `&source_config=${encodeURIComponent(state.item.source_config)}`
        : "";
      const patchPath = apiPath("/api/patch");
      const patchSeparator = patchPath.includes("?") ? "&" : "?";
      sourceImage.src = `${patchPath}${patchSeparator}well=${encodeURIComponent(state.item.well)}&timepoint=${encodeURIComponent(state.item.timepoint)}&x=${state.item.x_px}&y=${state.item.y_px}&size=${SIZE}&t=${Date.now()}${sourceQuery}`;
      $("notes").value = state.item.notes || "";
      setEditMode(false);
      updateDetails();
      drawCanvas();
      setMessage(boundaryMessage());
    } catch (error) { setMessage(error.message, true); }
  }

  function maskPosition(event) {
    const rect = canvas.getBoundingClientRect();
    return [
      Math.max(0, Math.min(SIZE - 1, Math.floor((event.clientX - rect.left) / rect.width * SIZE))),
      Math.max(0, Math.min(SIZE - 1, Math.floor((event.clientY - rect.top) / rect.height * SIZE))),
    ];
  }

  function paint(event) {
    if (!state.editing || !state.item) return;
    const [x, y] = maskPosition(event);
    const radius = Math.max(0, Math.floor(state.brushSize / 2));
    for (let yy = y - radius; yy <= y + radius; yy += 1) {
      for (let xx = x - radius; xx <= x + radius; xx += 1) {
        if (xx < 0 || xx >= SIZE || yy < 0 || yy >= SIZE) continue;
        if ((xx - x) ** 2 + (yy - y) ** 2 > radius ** 2 + 1) continue;
        state.currentMask[yy * SIZE + xx] = state.erasing ? 0 : 1;
      }
    }
    drawCanvas();
  }

  function fillHoles() {
    const outside = new Uint8Array(SIZE * SIZE);
    const stack = [];
    for (let x = 0; x < SIZE; x += 1) stack.push([x, 0], [x, SIZE - 1]);
    for (let y = 0; y < SIZE; y += 1) stack.push([0, y], [SIZE - 1, y]);
    while (stack.length) {
      const [x, y] = stack.pop();
      const index = y * SIZE + x;
      if (outside[index] || state.currentMask[index]) continue;
      outside[index] = 1;
      if (x > 0) stack.push([x - 1, y]);
      if (x < SIZE - 1) stack.push([x + 1, y]);
      if (y > 0) stack.push([x, y - 1]);
      if (y < SIZE - 1) stack.push([x, y + 1]);
    }
    for (let index = 0; index < state.currentMask.length; index += 1) {
      if (!state.currentMask[index] && !outside[index]) state.currentMask[index] = 1;
    }
    drawCanvas();
    setEditMode(true);
    setMessage("已填充从边界不可达的内部孔洞，请检查后保存。");
  }

  async function save(decisionOverride = "") {
    if (!state.item) return;
    const sameAsModel = masksEqual(state.currentMask, state.modelMask);
    const decision = decisionOverride || (sameAsModel ? "accepted" : "edited");
    if (decision === "edited" && !state.currentMask.some(Boolean)) {
      setMessage("编辑后的 Mask 不能为空；如果确实不是细胞，请使用“判为无有效细胞”。", true);
      return;
    }
    setMessage("保存中…");
    try {
      const result = await api("/api/mask-review-save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          round_id: state.roundId,
          candidate_id: state.item.candidate_id,
          decision,
          reviewed_mask_rle: encodeRle(state.currentMask),
          reviewer: "local_user",
          notes: $("notes").value || "",
        }),
      });
      setMessage(`已保存：${decision}，审核面积 ${result.reviewed_area_px} px²。`);
      await loadQueue(state.item.candidate_id);
    } catch (error) { setMessage(error.message, true); }
  }

  async function loadRounds() {
    const allRounds = await api("/api/mask-review-rounds");
    state.rounds = allRounds.filter((round) => state.reviewMode === "comparison"
      ? round.kind === "comparison"
      : round.kind !== "comparison");
    const select = $("round-select");
    select.innerHTML = state.rounds.map((round) => `<option value="${round.round_id}">${round.round_id} · 待审 ${round.pending_count}</option>`).join("");
    if (!state.rounds.length) {
      setMessage(state.reviewMode === "comparison" ? "尚未生成对比批次，请在上方选择留出样本并生成。" : "尚未生成审核批次，请先运行 scripts/run_p0_mask_review.py。", state.reviewMode !== "comparison");
      state.roundId = "";
      state.item = null;
      state.candidates = [];
      state.currentIndex = -1;
      updateDetails();
      renderSummary({ pending_count: 0, accepted_count: 0, edited_count: 0, rejected_count: 0 });
      renderQueue();
      drawCanvas();
      return;
    }
    state.roundId = state.rounds[0].round_id;
    select.value = state.roundId;
    await loadQueue();
  }

  $("round-select").addEventListener("change", async (event) => {
    state.roundId = event.target.value;
    state.currentIndex = -1;
    await loadQueue();
  });
  $("status-select").addEventListener("change", () => loadQueue());
  $("refresh-button").addEventListener("click", () => loadRounds().catch((error) => setMessage(error.message, true)));
  $("accept-old-button").addEventListener("click", () => { state.currentMask = state.oldModelMask.slice(); setEditMode(false); drawCanvas(); save("edited"); });
  $("accept-button").addEventListener("click", () => { state.currentMask = state.newModelMask.slice(); setEditMode(false); drawCanvas(); save("accepted"); });
  $("reject-button").addEventListener("click", () => { state.currentMask.fill(0); setEditMode(false); drawCanvas(); save("rejected"); });
  $("edit-button").addEventListener("click", () => setEditMode(!state.editing));
  $("brush-button").addEventListener("click", () => { state.erasing = false; $("brush-button").classList.add("active"); $("eraser-button").classList.remove("active"); setEditMode(true); });
  $("eraser-button").addEventListener("click", () => { state.erasing = true; $("eraser-button").classList.add("active"); $("brush-button").classList.remove("active"); setEditMode(true); });
  $("brush-size").addEventListener("input", (event) => { state.brushSize = Number(event.target.value); $("brush-size-value").textContent = String(state.brushSize); });
  $("fill-button").addEventListener("click", fillHoles);
  $("reset-button").addEventListener("click", () => { if (!state.item) return; state.currentMask = state.newModelMask.slice(); drawCanvas(); setMessage("已重置为新模型 Mask。"); });
  $("save-button").addEventListener("click", () => save());
  canvas.addEventListener("pointerdown", (event) => { if (!state.editing) return; state.drawing = true; canvas.setPointerCapture(event.pointerId); paint(event); });
  canvas.addEventListener("pointermove", (event) => { if (state.drawing) paint(event); });
  canvas.addEventListener("pointerup", () => { state.drawing = false; });
  canvas.addEventListener("pointercancel", () => { state.drawing = false; });
  document.addEventListener("keydown", (event) => {
    if (event.target && ["TEXTAREA", "INPUT", "SELECT"].includes(event.target.tagName)) return;
    if (event.key.toLowerCase() === "a") { event.preventDefault(); $("accept-button").click(); }
    else if (event.key.toLowerCase() === "e") { event.preventDefault(); $("edit-button").click(); }
    else if (event.key.toLowerCase() === "s") { event.preventDefault(); $("save-button").click(); }
    else if (event.key === "ArrowLeft") { event.preventDefault(); selectCandidate(state.currentIndex - 1); }
    else if (event.key === "ArrowRight") { event.preventDefault(); selectCandidate(state.currentIndex + 1); }
  });

  async function loadComparisonOptions() {
    state.comparisonOptions = await api("/api/mask-comparison-options");
    const checkpointSelect = $("comparison-checkpoint-select");
    checkpointSelect.innerHTML = (state.comparisonOptions.new_candidates || [])
      .map((item) => `<option value="${item.path}">${item.label}</option>`).join("");
    const sourceSelect = $("comparison-source-select");
    sourceSelect.innerHTML = (state.comparisonOptions.holdout_sources || [])
      .map((item) => `<option value="${item.source_config}" ${item.predictions_available && item.images_available ? "selected" : "disabled"}>${item.label} · ${item.candidate_count}候选</option>`).join("");
    $("comparison-description").textContent = state.comparisonOptions.new_checkpoint
      ? `旧模型为生产 checkpoint；新模型默认为 ${state.comparisonOptions.new_candidates?.[0]?.label || "最近候选"}。`
      : "没有找到可用的新模型候选。";
  }

  async function createComparisonRound() {
    const sourceConfigs = Array.from($("comparison-source-select").selectedOptions).map((option) => option.value);
    const newCheckpoint = $("comparison-checkpoint-select").value;
    if (!sourceConfigs.length || !newCheckpoint) {
      setMessage("请至少选择一个留出板位和一个新模型。", true);
      return;
    }
    const button = $("create-comparison-button");
    button.disabled = true;
    setMessage("正在对留出板位运行新旧模型，完成后会自动进入审核队列……");
    try {
      const result = await api("/api/mask-comparison-round", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ new_checkpoint: newCheckpoint, source_configs: sourceConfigs }),
      });
      state.roundId = result.round_id;
      await loadRounds();
      $("round-select").value = state.roundId;
      setMessage(`对比批次已生成：${result.round_id}，共 ${result.candidate_count} 个可审核对象。`);
    } catch (error) {
      setMessage(error.message, true);
    } finally {
      button.disabled = false;
    }
  }

  $("review-mode").addEventListener("change", async (event) => {
    state.reviewMode = event.target.value;
    state.currentIndex = -1;
    updateModeUi();
    if (state.reviewMode === "comparison") await loadComparisonOptions();
    await loadRounds();
  });
  $("create-comparison-button").addEventListener("click", () => createComparisonRound());

  updateModeUi();
  drawCanvas();
  loadRounds().catch((error) => setMessage(error.message, true));
})();
