const $ = id => document.getElementById(id);
let analysis = null;
const projectId = document.querySelector('meta[name="project-id"]')?.content || "";
let projectData = null;
let taskNameAutoValue = "";

if (projectId) {
  const multiplicityLink = $("multiplicityReviewLink");
  if (multiplicityLink) {
    multiplicityLink.href = `/single-doublet-review?project_id=${encodeURIComponent(projectId)}`;
  }
  const maskLink = $("maskReviewLink");
  if (maskLink) {
    maskLink.href = `/mask-review?project_id=${encodeURIComponent(projectId)}`;
  }
}

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

const taskStatusNames = {
  queued: { label: "排队中（未开始）", className: "queued" },
  running: { label: "执行中", className: "running" },
  completed: { label: "已完成", className: "completed" },
  error: { label: "执行失败", className: "error" },
  cancelled: { label: "已取消", className: "cancelled" },
};
const boardStageNames = {
  queued: "排队中",
  starting: "准备启动",
  day14_gate: "末点生长筛选",
  initialize_database: "初始化审核数据库",
  build_positive_only_t0_t2_manifest: "准备早期图像清单",
  build_cf_candidates: "生成候选细胞",
  dense_candidate_augmentation: "补充密集候选",
  morphology_inference: "单细胞形态识别",
  auto_annotation_round: "自动标注",
  multiplicity_inference: "单/粘连识别",
  integrated_round: "整合识别结果",
  v2_instance_segmentation: "实例分割",
  v2_temporal_evidence: "时序证据分析",
  early_well_screening: "孔级筛选",
  day7_localization_and_final_report: "生成最终报告",
  completed: "已完成",
  error: "失败",
};
const taskStatusHistory = new Map();
let taskPollTimer = null;
let taskPollInFlight = false;

function taskStatusInfo(task) {
  const status = String(task.status || "queued");
  return taskStatusNames[status] || { label: status, className: "unknown" };
}

function taskPercent(task) {
  const explicit = Number(task.progress_percent);
  if (Number.isFinite(explicit)) return Math.min(100, Math.max(0, explicit));
  const total = Number(task.progress_total || task.group_count || 0);
  const current = Number(task.progress_current || 0);
  return total > 0 ? Math.min(100, Math.max(0, current / total * 100)) : 0;
}

function taskProgressDetail(task) {
  const total = Number(task.progress_total || task.group_count || 0);
  const current = Number(task.progress_current || 0);
  if (total > 0) {
    return `${current.toLocaleString()} / ${total.toLocaleString()} 组`;
  }
  return task.progress_message || "等待执行";
}

function formatDuration(seconds) {
  const value = Math.max(0, Math.round(Number(seconds) || 0));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const secs = value % 60;
  if (hours) return `${hours}小时 ${String(minutes).padStart(2, "0")}分 ${String(secs).padStart(2, "0")}秒`;
  if (minutes) return `${minutes}分 ${String(secs).padStart(2, "0")}秒`;
  return `${secs}秒`;
}

function taskElapsedSeconds(task) {
  if (String(task.status) === "running" && task.started_at) {
    const started = Date.parse(task.started_at);
    if (Number.isFinite(started)) return Math.max(0, (Date.now() - started) / 1000);
  }
  return Number(task.elapsed_seconds || 0);
}

function renderBoardProgress(task) {
  const boards = Array.isArray(task.progress_boards) ? task.progress_boards : [];
  if (!boards.length) return "";
  const completed = boards.filter(board => String(board.status) === "completed").length;
  return `<details class="task-boards" ${String(task.status) === "running" ? "open" : ""}>
    <summary>板子进度：${completed}/${boards.length} 完成</summary>
    <div class="task-board-list">${boards.map(board => {
      const percent = Math.min(100, Math.max(0, Number(board.progress_percent || 0)));
      const elapsed = String(board.status) === "running" && board.started_at
        ? Math.max(0, (Date.now() - Date.parse(board.started_at)) / 1000)
        : Number(board.elapsed_seconds || 0);
      return `<div class="task-board">
        <strong class="task-board-title">${esc(board.board_id || board.slug || "板子")}</strong>
        <span class="task-board-stage">${esc(boardStageNames[board.stage] || board.stage || board.status || "排队中")}</span>
        <span class="task-board-message">${esc(board.message || "—")}<i class="task-board-progress"><i style="width:${percent}%"></i></i></span>
        <span class="task-board-time">${formatDuration(elapsed)}<br>${Math.round(percent)}%</span>
      </div>`;
    }).join("")}</div>
  </details>`;
}

function taskActions(task) {
  const status = String(task.status || "queued");
  const buttons = [];
  if (status === "queued") {
    buttons.push(`<button class="task-action primary" type="button" data-task-action="start" data-task-id="${esc(task.task_id)}">开始计算</button>`);
  }
  if (status === "queued" || status === "running") {
    buttons.push(`<button class="task-action secondary" type="button" data-task-action="cancel" data-task-id="${esc(task.task_id)}">取消</button>`);
  }
  if (status !== "completed" && status !== "running") {
    buttons.push(`<button class="task-action danger" type="button" data-task-action="delete" data-task-id="${esc(task.task_id)}">删除</button>`);
  }
  return buttons.length ? `<div class="task-actions">${buttons.join("")}</div>` : "";
}

function renderTask(task) {
  const status = taskStatusInfo(task);
  const percent = taskPercent(task);
  return `<div class="task" data-task-id="${esc(task.task_id)}">
    <div class="task-title"><strong>${esc(task.name)}</strong><small>${esc(task.task_id)}</small></div>
    <div class="task-path" title="${esc(task.path)}">${esc(task.path)}</div>
    <div class="task-state">
      <div class="task-state-line"><span class="status ${status.className}">${status.label}</span><b>${Math.round(percent)}%</b></div>
      <div class="task-progress" aria-label="任务进度"><i style="width:${percent}%"></i></div>
      <small>${esc(task.progress_message || taskProgressDetail(task))}</small>
    </div>
    <small class="task-meta">创建人：${esc(task.created_by || "未填写")} · 总体进度：${taskProgressDetail(task)} · 累计运行：${formatDuration(taskElapsedSeconds(task))}<br>${new Date(task.created_at).toLocaleString()}</small>
    ${taskActions(task)}
    ${renderBoardProgress(task)}
  </div>`;
}

async function handleTaskAction(event) {
  const button = event.currentTarget;
  const action = button.dataset.taskAction;
  const taskId = button.dataset.taskId;
  if (!action || !taskId || button.disabled) return;
  if (action === "cancel" && !window.confirm("确定取消这个任务吗？")) return;
  if (action === "delete" && !window.confirm("只会删除任务记录，不会删除原始数据。确定删除吗？")) return;
  button.disabled = true;
  try {
    const method = action === "delete" ? "DELETE" : "POST";
    const endpoint = action === "delete"
      ? `/api/project/tasks/${encodeURIComponent(taskId)}`
      : `/api/project/tasks/${encodeURIComponent(taskId)}/${action}`;
    await api(endpoint, { method });
    toast(action === "start" ? "任务已开始，等待执行器接管" : action === "cancel" ? "任务已取消" : "任务已删除");
    await loadTasks();
  } catch (error) {
    toast(`任务操作失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
}

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
  projectData = data;
  $("projectName").textContent = data.project_name || data.project_id;
  $("projectMeta").textContent = `${dateRange(data)} · 创建人：${data.created_by || "—"}`;

  const plates = data.plates || [];
  $("deleteProjectButton").hidden = plates.length !== 0;
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
    const target = ready
      ? `/projects/${encodeURIComponent(projectId)}/plates/${encodeURIComponent(plate.slug)}/`
      : "#";
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
    startTaskPolling();
  } catch (error) {
    toast(`加载失败：${error.message}`);
  }
}

async function loadTasks({ silent = false } = {}) {
  if (taskPollInFlight) return;
  taskPollInFlight = true;
  try {
    const query = projectId ? `?project_id=${encodeURIComponent(projectId)}` : "";
    const tasks = await api(`/api/project/tasks${query}`);
    tasks.forEach(task => {
      const id = String(task.task_id || "");
      const previous = taskStatusHistory.get(id);
      const current = String(task.status || "queued");
      if (previous && previous !== current && current !== "queued") {
        toast(`${task.name || "任务"}：${taskStatusInfo(task).label}`);
      }
      taskStatusHistory.set(id, current);
    });
    $("taskRows").innerHTML = tasks.length
      ? tasks.slice().reverse().map(renderTask).join("")
      : `<div class="empty">暂无任务</div>`;
    document.querySelectorAll("[data-task-action]").forEach(button => {
      button.addEventListener("click", handleTaskAction);
    });
  } catch (error) {
    if (!silent) throw error;
  } finally {
    taskPollInFlight = false;
  }
}

function startTaskPolling() {
  if (taskPollTimer) clearInterval(taskPollTimer);
  taskPollTimer = setInterval(() => {
    if (document.visibilityState === "hidden") return;
    loadTasks({ silent: true });
  }, 2500);
}

function showAnalysis(value) {
  analysis = value;
  applyDefaultTaskName(value.folder_name || folderNameFromPath($("folderPath").value));
  $("analysisResult").textContent = `已解析：${value.group_count} 个板组，${value.session_count} 个时间点文件夹\n时间点：${value.timepoint_labels.join(", ")} · 日龄：${value.day_labels.join(", ")}\n完整96孔组：${value.complete_groups}`;
  $("queueButton").disabled = false;
}

function closeTaskDialog() {
  const dialog = $("taskDialog");
  if (dialog?.open) dialog.close("cancel");
}

function folderNameFromPath(value) {
  const parts = String(value || "").replace(/[\\\\/]+$/, "").split(/[\\\\/]/).filter(Boolean);
  if (!parts.length) return "";
  const last = parts[parts.length - 1];
  return last.toLowerCase() === "sessions.idx" ? (parts[parts.length - 2] || last) : last;
}

function applyDefaultTaskName(value) {
  const name = String(value || "").trim();
  const input = $("taskName");
  if (!name || !input) return;
  if (!input.dataset.edited || !input.value.trim() || input.value === taskNameAutoValue) {
    input.value = name;
    taskNameAutoValue = name;
  }
}

function closeProjectNameDialog() {
  const dialog = $("projectNameDialog");
  if (dialog?.open) dialog.close("cancel");
}

async function saveProjectName() {
  const input = $("projectNameInput");
  const name = input.value.trim();
  if (!name) {
    toast("项目名称不能为空");
    return;
  }
  try {
    await api(`/api/project/${encodeURIComponent(projectId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ project_name: name }),
    });
    closeProjectNameDialog();
    toast("项目名称已修改");
    await loadProject();
  } catch (error) {
    toast(`修改项目名称失败：${error.message}`);
  }
}

async function deleteEmptyProject() {
  if (!projectData || Number(projectData.plate_count || 0) !== 0) {
    toast("只有没有板子的空项目可以删除");
    return;
  }
  if (!window.confirm("该项目没有板子。确认删除项目记录吗？原始数据文件夹不会被删除。")) return;
  try {
    await api(`/api/project/${encodeURIComponent(projectId)}`, { method: "DELETE" });
    window.location.href = "/";
  } catch (error) {
    toast(`删除项目失败：${error.message}`);
  }
}

$("refreshButton").addEventListener("click", loadProject);
$("renameProjectButton").addEventListener("click", () => {
  $("projectNameInput").value = projectData?.project_name || $("projectName").textContent || "";
  $("projectNameDialog").showModal();
});
$("deleteProjectButton").addEventListener("click", deleteEmptyProject);
// The dashboard keeps the task entry point as a normal link so it also works
// when JavaScript is still loading.  Older cached dashboard pages did expose
// a button with this id, so keep the listener guarded for compatibility.
const newTaskButton = $("newTaskButton");
if (newTaskButton) {
  newTaskButton.addEventListener("click", () => { window.location.href = "/?new-task=1"; });
}
$("closeTaskButton").addEventListener("click", closeTaskDialog);
$("cancelTaskButton").addEventListener("click", closeTaskDialog);
$("closeProjectNameButton").addEventListener("click", closeProjectNameDialog);
$("cancelProjectNameButton").addEventListener("click", closeProjectNameDialog);
$("saveProjectNameButton").addEventListener("click", saveProjectName);
$("taskName").addEventListener("input", () => { $("taskName").dataset.edited = "1"; });
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
      body: JSON.stringify({
        name: $("taskName").value.trim(),
        created_by: $("createdBy").value.trim(),
        path: $("folderPath").value.trim(),
      }),
    });
    $("taskDialog").close();
    toast(`已加入任务队列：${task.name}`);
    await loadTasks();
  } catch (error) {
    toast(`加入队列失败：${error.message}`);
  }
});

loadProject();
