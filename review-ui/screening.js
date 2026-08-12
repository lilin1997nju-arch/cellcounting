const state = { wells: [], selected: null };
const $ = id => document.getElementById(id);

const statusText = {
  single_active: "单细胞有活性",
  single_not_divided: "T0-T2未分裂",
  multi_origin: "多细胞来源",
  no_cell_growth: "无明显生长",
  t0_missing_late_cells: "T0缺失但后期出现细胞",
  positive_control: "阳性对照",
  single_growth_unconfirmed: "T0-T2未分裂",
  missing_t0_or_late_object: "T0缺失但后期出现细胞",
  no_cell: "无明显生长",
  ambiguous: "T0缺失但后期出现细胞"
};
const growthDecisionText = {
  obvious_growth: "明显生长",
  no_growth: "无明显生长",
  uncertain: "无法判断",
  pending: "待标记"
};
const statusClass = status => ({
  single_active: "good",
  single_not_divided: "single",
  multi_origin: "multi",
  no_cell_growth: "no-growth",
  t0_missing_late_cells: "issue",
  single_growth_unconfirmed: "single",
  missing_t0_or_late_object: "issue",
  no_cell: "no-growth",
  ambiguous: "issue"
}[status] || "issue");

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function msg(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function renderSummary() {
  const high = state.wells.filter(well => well.high_confidence_single_active).length;
  const skipped = state.wells.filter(well => well.skip_deep_search).length;
  const pending = state.wells.filter(well => well.late_growth_status === "pending").length;
  const required = state.wells.filter(well => well.deep_search_required).length;
  $("summary").innerHTML = `
    <div class="metric"><b>${high}</b><span>单细胞有活性</span></div>
    <div class="metric"><b>${skipped}</b><span>晚期无生长，跳过深检</span></div>
    <div class="metric"><b>${pending}</b><span>T3/T4待快速标记</span></div>
    <div class="metric"><b>${required}</b><span>保留深度检索资格</span></div>`;
}

function renderPlate() {
  const plate = $("plate");
  plate.innerHTML = '<div></div>' + Array.from(
    { length: 12 }, (_, index) => `<div class="axis">${index + 1}</div>`
  ).join("");
  for (const row of "ABCDEFGH") {
    plate.insertAdjacentHTML("beforeend", `<div class="axis">${row}</div>`);
    for (let column = 1; column <= 12; column += 1) {
      const well = `${row}${column}`;
      const item = state.wells.find(value => value.well === well);
      if (!item) {
        plate.insertAdjacentHTML("beforeend", "<div></div>");
        continue;
      }
      const button = document.createElement("button");
      button.className = `well ${statusClass(item.screening_status)} ${item.review_decision || ""}`;
      button.textContent = well;
      button.title = `${statusText[item.screening_status] || item.screening_status}；晚期：${growthDecisionText[item.late_growth_status] || item.late_growth_status}`;
      button.onclick = () => selectWell(well);
      plate.appendChild(button);
    }
  }
}

function imagePair(well, timepoint, units) {
  return `<article class="tp-card">
    <div class="tp-title"><span>${timepoint}</span><span>${units} 个细胞单位</span></div>
    <div class="image-pair">
      <figure><img loading="lazy" src="/api/report-image?well=${well}&timepoint=${timepoint}&view=whole&max_size=900"><figcaption>整体视野（白框为局部位置）</figcaption></figure>
      <figure><img loading="lazy" src="/api/report-image?well=${well}&timepoint=${timepoint}&view=local&max_size=900"><figcaption>局部原图（不绘制细胞圈）</figcaption></figure>
    </div>
  </article>`;
}

function confirmationCard(well, timepoint) {
  const key = timepoint.toLowerCase();
  if (!well[`${key}_available`]) return "";
  const decision = well[`${key}_growth_decision`] || "pending";
  const choices = [
    ["obvious_growth", "明显生长"],
    ["no_growth", "无明显生长"],
    ["uncertain", "无法判断"]
  ].map(([value, label]) => `
    <button class="growth-choice ${decision === value ? "active" : ""}"
      data-timepoint="${timepoint}" data-decision="${value}">${label}</button>`
  ).join("");
  return `<article class="tp-card late-growth-card">
    <div class="tp-title"><span>${timepoint} 生长确认</span><span>${growthDecisionText[decision]}</span></div>
    <div class="image-pair">
      <figure><img loading="lazy" src="/api/report-image?well=${well.well}&timepoint=${timepoint}&view=whole&max_size=1200"><figcaption>整体视野（用于判断孔内是否有明显生长）</figcaption></figure>
      <figure><img loading="lazy" src="/api/report-image?well=${well.well}&timepoint=${timepoint}&view=local&max_size=1200"><figcaption>局部无标记视野</figcaption></figure>
    </div>
    <div class="growth-actions">${choices}</div>
  </article>`;
}

function gateBanner(well) {
  const mapping = {
    no_growth: ["stop", "T3/T4无明显细胞生长：该孔已跳过深度检索，报告标记为无细胞生长。"],
    obvious_growth: ["go", "晚期图像存在明显细胞生长：该孔保留深度检索资格。"],
    uncertain: ["pending", "晚期生长无法判断：为避免漏检，该孔仍保留深度检索资格。"],
    pending: ["pending", "请快速标记所有可用的T3/T4图像；完成前不会跳过计算。"],
    unavailable: ["pending", "该孔没有可用的T3/T4图像，不能使用晚期生长门控。"]
  };
  const [kind, text] = mapping[well.late_growth_status] || mapping.pending;
  return `<div class="growth-gate ${kind}">${text}</div>`;
}

function bindGrowthActions() {
  document.querySelectorAll(".growth-choice").forEach(button => {
    button.onclick = () => saveLateGrowthDecision(
      button.dataset.timepoint,
      button.dataset.decision
    );
  });
}

function selectWell(wellName) {
  state.selected = state.wells.find(value => value.well === wellName);
  if (!state.selected) return;
  document.querySelectorAll(".well").forEach(node => {
    node.classList.toggle("selected", node.textContent === wellName);
  });
  const well = state.selected;
  $("wellDetail").classList.remove("hidden");
  $("wellTitle").textContent = wellName;
  $("wellVerdict").textContent = `${statusText[well.screening_status]}；T0/T1/T2细胞单位：${well.t0_cell_units}/${well.t1_cell_units}/${well.t2_cell_units}${well.has_debris ? "；检测到杂质" : ""}`;
  $("openAudit").href = `/auto-review?well=${wellName}`;
  const confirmations = confirmationCard(well, "T3") + confirmationCard(well, "T4");
  $("timepointReport").innerHTML = gateBanner(well)
    + imagePair(wellName, "T0", well.t0_cell_units)
    + imagePair(wellName, "T1", well.t1_cell_units)
    + imagePair(wellName, "T2", well.t2_cell_units)
    + (confirmations ? `<div class="confirmations">${confirmations}</div>` : "");
  bindGrowthActions();
  $("wellDetail").scrollIntoView({ behavior: "smooth", block: "start" });
  msg(`${wellName} 已载入；T3/T4只需判断是否存在明显细胞生长`);
}

async function saveLateGrowthDecision(timepoint, decision) {
  if (!state.selected) return;
  const well = state.selected.well;
  msg(`正在保存 ${well} ${timepoint} 生长标记…`);
  try {
    await api("/api/late-growth-review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ well, timepoint, decision, reviewer: "local_user" })
    });
    await load(well);
    msg(`${well} ${timepoint} 已标记为“${growthDecisionText[decision]}”`);
  } catch (error) {
    msg(`保存失败：${error.message}`, true);
  }
}

async function saveDecision(decision) {
  if (!state.selected) return;
  const well = state.selected.well;
  msg(`正在保存 ${well}…`);
  await api("/api/screening-review", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ well, decision, reviewer: "local_user" })
  });
  await load(well);
  msg(`${well} 已${decision === "approved" ? "审核通过" : "标记为判定不正确"}`);
}

function reportCard(well) {
  const timepoints = ["T0", "T1", "T2", ...(well.t3_available ? ["T3"] : []), ...(well.t4_available ? ["T4"] : [])];
  const figures = timepoints.map(timepoint => `
    <figure><img loading="lazy" src="/api/report-image?well=${well.well}&timepoint=${timepoint}&view=whole&max_size=700"><figcaption>${timepoint} 整体视野（白框为局部位置）</figcaption></figure>
    <figure><img loading="lazy" src="/api/report-image?well=${well.well}&timepoint=${timepoint}&view=local&max_size=700"><figcaption>${timepoint} 局部无标记视野</figcaption></figure>`
  ).join("");
  return `<article class="report-well"><h3>${well.well} <small>${well.review_decision === "approved" ? "已审核通过" : "模型预选·待孔级审核"}</small></h3><div class="report-images">${figures}</div></article>`;
}

function renderReport() {
  const approvedOnly = $("approvedOnly").checked;
  const rows = state.wells.filter(well =>
    well.high_confidence_single_active
    && !well.skip_deep_search
    && well.review_decision !== "rejected"
    && (!approvedOnly || well.review_decision === "approved")
  );
  $("reportList").innerHTML = rows.length
    ? rows.map(reportCard).join("")
    : "<p>当前筛选条件下暂无孔位。</p>";
}

function selectNextLatePending() {
  const currentIndex = state.selected
    ? state.wells.findIndex(well => well.well === state.selected.well)
    : -1;
  const ordered = [
    ...state.wells.slice(currentIndex + 1),
    ...state.wells.slice(0, currentIndex + 1)
  ];
  const next = ordered.find(well =>
    (well.t3_available || well.t4_available)
    && well.late_growth_status === "pending"
  );
  if (next) selectWell(next.well);
  else msg("所有可用T3/T4图像均已完成快速标记");
}

async function load(selectedWell = null) {
  state.wells = await api("/api/screening-wells");
  renderSummary();
  renderPlate();
  renderReport();
  if (selectedWell) selectWell(selectedWell);
  else msg(`已载入 ${state.wells.length} 个孔位`);
}

$("approveButton").onclick = () => saveDecision("approved");
$("rejectButton").onclick = () => saveDecision("rejected");
$("nextLateButton").onclick = selectNextLatePending;
$("approvedOnly").onchange = renderReport;
load().catch(error => msg(`载入失败：${error.message}`, true));
