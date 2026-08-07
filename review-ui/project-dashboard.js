const $ = id => document.getElementById(id);
let analysis = null;
const projectId = document.querySelector('meta[name="project-id"]')?.content || "";

const toast = text => {
  $("toast").textContent = text;
  $("toast").classList.add("show");
  setTimeout(() => $("toast").classList.remove("show"), 2400);
};

const esc = value => String(value ?? "").replace(/[&<>\"]/g, character => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  "\"": "&quot;",
}[character]));

const count = value => Number(value || 0);

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function summaryCard(label, value) {
  return `<div class="summary-card"><span>${label}</span><strong>${count(value).toLocaleString()}</strong></div>`;
}

function dateOnly(value) {
  if (!value) return "—";
  const match = String(value).match(/^\d{4}-\d{2}-\d{2}/);
  if (match) return match[0];
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "—" : date.toLocaleDateString("zh-CN");
}

function dateRange(data) {
  const start = dateOnly(data.detection_start_date);
  const end = dateOnly(data.detection_end_date);
  if (start === "—" && end === "—") return "检测日期未记录";
  if (start === end || end === "—") return `检测日期：${start}`;
  if (start === "—") return `检测日期：${end}`;
  return `检测日期：${start} ～ ${end}`;
}

function plateStatus(plate) {
  if (plate.review_complete) return { label: "已审核", className: "reviewed" };
  if (plate.status === "completed") return { label: "已完成识别", className: "recognized" };
  if (plate.status === "running") return { label: "识别中", className: "running" };
  if (plate.status === "error") return { label: "失败", className: "error" };
  return { label: "待识别", className: "queued" };
}

function reviewText(plate) {
  if (!plate.review_data_available) return "已审核 — / —";
  return `已审核 ${count(plate.reviewed_well_count)}/${count(plate.reviewable_well_count)}`;
}

function resultText(plate) {
  const counts = plate.category_counts || {};
  return `<span class="result-summary">单细胞 ${count(counts.single_cell_origin).toLocaleString()} · 多细胞 ${count(counts.multi_cell_origin).toLocaleString()} · 待确定 ${count(counts.undetermined).toLocaleString()}</span>`;
}

function renderProject(data) {
  $("projectName").textContent = data.project_name || data.project_id;
  $("projectMeta").textContent = `${dateRange(data)} · 创建人：${data.created_by || "—"}`;

  const plates = data.plates || [];
  const recognized = count(data.recognized_plate_count ?? plates.filter(plate => plate.status === "completed").length);
  const reviewed = count(data.reviewed_plate_count ?? plates.filter(plate => plate.review_complete).length);
  const mounted = data.mounted_plates || [];
  $("mountedMeta").textContent = `已完成识别 ${recognized}/${plates.length} · 已审核 ${reviewed}/${plates.length}`;

  const counts = data.category_counts || {};
  $("summaryCards").innerHTML = [
    summaryCard("无明显生长", counts.no_obvious_growth),
    summaryCard("单细胞来源孔", counts.single_cell_origin),
    summaryCard("多细胞来源孔", counts.multi_cell_origin),
    summaryCard("待确定", counts.undetermined),
    summaryCard("阳性对照", counts.positive_control),
  ].join("");

  $("plateRows").innerHTML = plates.length ? plates.map(plate => {
    const ready = mounted.includes(plate.slug);
    const target = ready ? `/plates/${encodeURIComponent(plate.slug)}/` : "#";
    const status = plateStatus(plate);
    const countsForPlate = plate.category_counts || {};
    return `<tr class="${ready ? "clickable" : ""}" ${ready ? `data-href="${esc(target)}"` : ""}>
      <td><span class="plate-link">${esc(plate.board_id || plate.group_id)}</span><br><small>${esc(plate.group_id)}</small></td>
      <td><span class="status ${status.className}">${status.label}</span></td>
      <td><span class="review-progress">${reviewText(plate)}</span><small>${count(plate.reviewed_object_count).toLocaleString()}/${count(plate.reviewable_object_count).toLocaleString()} 个对象</small></td>
      <td>${resultText({ category_counts: countsForPlate })}</td>
      <td>${count(plate.well_count).toLocaleString()}</td>
      <td>${plate.elapsed_seconds ? `${Number(plate.elapsed_seconds).toFixed(1)} s` : "—"}</td>
      <td>${ready ? `<a class="plate-open" href="${esc(target)}">进入审核 →</a>` : ""}</td>
    </tr>`;
  }).join("") : `<tr><td colspan="7" class="empty">暂无板子</td></tr>`;

  document.querySelectorAll("tr[data-href]").forEach(row => row.addEventListener("click", event => {
    if (event.target.closest("a")) return;
    window.location.href = row.dataset.href;
  }));
}

async function loadProject() {
  try {
    renderProject(await api(`/api/project${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`));
    await loadTasks();
  } catch (error) {
    toast(`加载失败：${error.message}`);
  }
}

async function loadTasks() {
  const tasks = await api("/api/project/tasks");
  $("taskRows").innerHTML = tasks.length
    ? tasks.slice().reverse().map(task => `<div class="task"><strong>${esc(task.name)}</strong><span>${esc(task.path)}</span><span class="status">${esc(task.status)}</span><small>${task.group_count || 0} 组 · ${new Date(task.created_at).toLocaleString()}</small></div>`).join("")
    : `<div class="empty">暂无任务</div>`;
}

function showAnalysis(value) {
  analysis = value;
  $("analysisResult").textContent = `已解析：${value.group_count} 个板组，${value.session_count} 个时间点文件夹\n时间点：${value.timepoint_labels.join(", ")} · 日龄：${value.day_labels.join(", ")}\n完整96孔组：${value.complete_groups}`;
  $("queueButton").disabled = false;
}

function closeTaskDialog() {
  const dialog = $("taskDialog");
  if (dialog?.open) dialog.close("cancel");
}

$("refreshButton").addEventListener("click", loadProject);
// The dashboard keeps the task entry point as a normal link so it also works
// when JavaScript is still loading.  Older cached dashboard pages did expose
// a button with this id, so keep the listener guarded for compatibility.
const newTaskButton = $("newTaskButton");
if (newTaskButton) {
  newTaskButton.addEventListener("click", () => { window.location.href = "/?new-task=1"; });
}
$("closeTaskButton").addEventListener("click", closeTaskDialog);
$("cancelTaskButton").addEventListener("click", closeTaskDialog);
$("browseButton").addEventListener("click", async () => {
  $("analysisResult").textContent = "正在打开文件夹选择器…";
  try {
    const result = await api("/api/project/browse-folder", { method: "POST" });
    if (result.path) {
      $("folderPath").value = result.path;
      $("analysisResult").textContent = "已选择文件夹，请点击解析数据。";
    } else {
      $("analysisResult").textContent = result.error || "未选择文件夹；也可以直接输入文件夹路径。";
    }
  } catch (error) {
    $("analysisResult").textContent = `选择文件夹失败：${error.message}`;
  }
});
$("analyzeButton").addEventListener("click", async () => {
  try {
    showAnalysis(await api("/api/project/analyze-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("folderPath").value.trim() }),
    }));
  } catch (error) {
    $("analysisResult").textContent = `解析失败：${error.message}`;
    $("queueButton").disabled = true;
  }
});
$("queueButton").addEventListener("click", async () => {
  try {
    const task = await api("/api/project/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("taskName").value.trim(), path: $("folderPath").value.trim() }),
    });
    $("taskDialog").close();
    toast(`已加入任务队列：${task.name}`);
    await loadTasks();
  } catch (error) {
    toast(`加入队列失败：${error.message}`);
  }
});

loadProject();
