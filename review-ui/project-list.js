const $ = id => document.getElementById(id);
let analysis = null;
let taskNameAutoValue = "";
const PROJECTS_PER_PAGE = 10;
let allProjects = [];
let projectPage = 1;
let projectSearchTerm = "";
let projectTotal = 0;
let projectTotalPages = 1;
let projectAggregate = null;
let projectSearchTimer = null;
let projectPollTimer = null;
let reviewPlatformMode = false;
let reviewPlatformStatus = null;
let platformModeChecked = false;

const esc = value => String(value ?? "").replace(/[&<>\"]/g, character => ({
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  "\"": "&quot;",
}[character]));

const toast = text => {
  $("toast").textContent = text;
  $("toast").classList.add("show");
  setTimeout(() => $("toast").classList.remove("show"), 2600);
};

async function api(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

async function configurePlatformMode() {
  if (platformModeChecked) return;
  platformModeChecked = true;
  const response = await fetch("/api/review-platform/status");
  if (response.status === 404) {
    if (new URLSearchParams(location.search).get("new-task") === "1") {
      openNewTaskDialog();
    }
    return;
  }
  if (!response.ok) throw new Error(await response.text());
  reviewPlatformStatus = await response.json();
  reviewPlatformMode = Boolean(reviewPlatformStatus.enabled);
  if (!reviewPlatformMode) return;
  document.body.classList.add("review-platform-mode");
  document.title = "Cell Vision 离线审核平台";
  $("hubSubtitle").textContent = "导入 .cvreview 审核数据，可从项目列表进入各个板的完整审核界面。";
  $("projectSectionHelp").textContent = "审核数据可以放在任意路径；导入多个项目后会统一列在这里。";
  $("newTaskButton").textContent = "导入审核数据";
  $("taskQueuePanel").hidden = true;
  closeTaskDialog();
}

function reviewPlatformMeta(defaultText) {
  if (!reviewPlatformMode) return defaultText;
  const missing = Number(reviewPlatformStatus?.missing_count || 0);
  const available = Number(reviewPlatformStatus?.available_count || 0);
  return missing
    ? `已导入 ${available} 个 · ${missing} 个原路径已失效，请重新导入`
    : `已导入 ${available} 个审核项目`;
}

function count(value) {
  return Number(value || 0);
}

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
  final_report: "生成最终报告",
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
  const selected = (task.selected_timepoint_labels || []).join(", ");
  return `<div class="task" data-task-id="${esc(task.task_id)}">
    <div class="task-title"><strong>${esc(task.name)}</strong><small>${esc(task.task_id)}</small></div>
    <div class="task-path" title="${esc(task.path)}">${esc(task.path)}</div>
    <div class="task-state">
      <div class="task-state-line"><span class="status ${status.className}">${status.label}</span><b>${Math.round(percent)}%</b></div>
      <div class="task-progress" aria-label="任务进度"><i style="width:${percent}%"></i></div>
      <small>${esc(task.progress_message || taskProgressDetail(task))}</small>
    </div>
    <small class="task-meta">创建人：${esc(task.created_by || "未填写")} · 总体进度：${taskProgressDetail(task)} · 累计运行：${formatDuration(taskElapsedSeconds(task))}<br>已选 ${esc(selected || "—")} · 末点 ${esc(task.endpoint_day_label || "—")}</small>
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

function dateRange(item) {
  const start = dateOnly(item.detection_start_date);
  const end = dateOnly(item.detection_end_date);
  if (start === "—" && end === "—") return "—";
  if (start === end || end === "—") return start;
  if (start === "—") return end;
  return `${start} ～ ${end}`;
}

function projectLocation(item) {
  if (reviewPlatformMode && item.manifest_path) {
    return String(item.manifest_path).replace(/[\\/]project[\\/]project\.json$/i, "");
  }
  return item.root || "未记录数据根目录";
}

function projectStatus(item) {
  const total = count(item.plate_count);
  const recognized = count(item.recognized_plate_count ?? item.completed_plate_count);
  const reviewed = count(item.reviewed_plate_count);
  if (total > 0 && reviewed >= total) return { label: "已审核", className: "reviewed" };
  if (total > 0 && recognized >= total) return { label: "已完成识别", className: "recognized" };
  if (recognized > 0) return { label: "识别中", className: "processing" };
  return { label: "待识别", className: "pending" };
}

function filteredProjects() {
  const term = projectSearchTerm.trim().toLocaleLowerCase();
  if (!term) return allProjects;
  return allProjects.filter(item => [
    item.project_name,
    item.project_id,
    item.root,
    item.created_by,
    item.manifest_path,
  ].some(value => String(value || "").toLocaleLowerCase().includes(term)));
}

function updateProjectPagination(totalItems = projectTotal, totalPages = projectTotalPages) {
  const pagination = $("projectPagination");
  if (!pagination) return;
  pagination.hidden = totalItems <= PROJECTS_PER_PAGE;
  $("projectPageInfo").textContent = `第 ${projectPage} / ${totalPages} 页 · 共 ${totalItems} 个`;
  $("projectPrev").disabled = projectPage <= 1;
  $("projectNext").disabled = projectPage >= totalPages;
}

function renderProjects(items, meta = {}) {
  allProjects = Array.isArray(items) ? items : [];
  const filtered = filteredProjects();
  const serverPaged = Boolean(meta.serverPaged);
  const totalItems = serverPaged ? Number(meta.total || 0) : filtered.length;
  const totalPages = serverPaged
    ? Math.max(1, Number(meta.pages || 1))
    : Math.max(1, Math.ceil(filtered.length / PROJECTS_PER_PAGE));
  projectTotal = totalItems;
  projectTotalPages = totalPages;
  projectAggregate = serverPaged ? (meta.aggregate || null) : null;
  projectPage = Math.min(Math.max(1, projectPage), totalPages);
  const start = (projectPage - 1) * PROJECTS_PER_PAGE;
  items = serverPaged ? allProjects : filtered.slice(start, start + PROJECTS_PER_PAGE);
  const rows = $("projectRows");
  let boards = Number(projectAggregate?.plate_count || 0);
  let recognized = Number(projectAggregate?.recognized_plate_count || 0);
  if (!serverPaged) {
    boards = 0;
    recognized = 0;
    filtered.forEach(item => {
      boards += count(item.plate_count);
      recognized += count(item.recognized_plate_count ?? item.completed_plate_count);
    });
  }
  $("projectCount").textContent = `${totalItems} 个项目`;
  $("projectFilterMeta").textContent = projectSearchTerm.trim()
    ? `搜索到 ${totalItems} 个项目`
    : reviewPlatformMeta(`每页最多 ${PROJECTS_PER_PAGE} 个`);

  $("summaryCards").innerHTML = [
    summaryCard("项目", totalItems),
    summaryCard("板子", boards),
    summaryCard("已完成识别板子", recognized),
    summaryCard("待处理板子", Math.max(0, boards - recognized)),
  ].join("");

  if (!items.length) {
    rows.innerHTML = `<tr><td colspan="6" class="empty">${reviewPlatformMode ? "暂无审核项目，请点击“导入审核数据”并选择 .cvreview 文件夹。" : "暂无项目，请新建任务。"}</td></tr>`;
    updateProjectPagination(totalItems, totalPages);
    return;
  }

  rows.innerHTML = items.map(item => {
    const total = count(item.plate_count);
    const recognizedCount = count(item.recognized_plate_count ?? item.completed_plate_count);
    const reviewedCount = count(item.reviewed_plate_count);
    const status = projectStatus(item);
    const categories = item.category_counts || {};
    const singleCell = count(item.single_cell_origin_well_count ?? categories.single_cell_origin);
    const multiCell = count(categories.multi_cell_origin);
    const undetermined = count(categories.undetermined);
    const deleteAction = !reviewPlatformMode && total === 0
      ? `<button class="project-delete danger" type="button" data-project-delete="${esc(item.project_id)}" data-project-name="${esc(item.project_name || "空项目")}">删除空项目</button>`
      : "";
    return `<tr class="project-row" data-url="${esc(item.detail_url)}">
      <td><strong class="project-name">${esc(item.project_name)}</strong><span class="project-subline">${esc(projectLocation(item))}</span></td>
      <td>${dateRange(item)}</td>
      <td>${esc(item.created_by || "—")}</td>
      <td><span class="project-status ${status.className}">${status.label}</span><span class="project-status-detail">已完成识别 ${recognizedCount}/${total} · 已审核 ${reviewedCount}/${total}</span></td>
      <td><span class="project-result">单细胞来源孔：${singleCell.toLocaleString()}</span><span class="project-subline">多细胞来源 ${multiCell.toLocaleString()} · 待确定 ${undetermined.toLocaleString()}</span></td>
      <td class="project-actions"><a class="project-open" href="${esc(item.detail_url)}">进入项目 →</a>${deleteAction}</td>
    </tr>`;
  }).join("");

  rows.querySelectorAll("[data-project-delete]").forEach(button => button.addEventListener("click", deleteEmptyProjectFromList));
  rows.querySelectorAll("tr[data-url]").forEach(row => row.addEventListener("click", event => {
    if (event.target.closest("a, button")) return;
    window.location.href = row.dataset.url;
  }));
  updateProjectPagination(totalItems, totalPages);
}

async function deleteEmptyProjectFromList(event) {
  const button = event.currentTarget;
  const projectId = button.dataset.projectDelete;
  const projectName = button.dataset.projectName || "空项目";
  if (!projectId || button.disabled) return;
  if (!window.confirm(`项目“${projectName}”没有板子，确认删除项目记录吗？原始数据文件夹不会被删除。`)) return;
  button.disabled = true;
  try {
    await api(`/api/project/${encodeURIComponent(projectId)}`, { method: "DELETE" });
    toast("空项目已删除");
    await load();
  } catch (error) {
    toast(`删除项目失败：${error.message}`);
    button.disabled = false;
  }
}

async function loadProjects(page = projectPage, { silent = false } = {}) {
  const params = new URLSearchParams({
    q: projectSearchTerm.trim(),
    page: String(Math.max(1, page)),
    page_size: String(PROJECTS_PER_PAGE),
  });
  const result = await api(`/api/projects?${params.toString()}`);
  projectPage = Number(result.page || page || 1);
  renderProjects(result.items || [], { ...result, serverPaged: true });
}

async function load() {
  try {
    await configurePlatformMode();
    if (reviewPlatformMode) reviewPlatformStatus = await api("/api/review-platform/status");
    await loadProjects(1);
    if (!reviewPlatformMode) await loadTasks();
    if (projectPollTimer) clearInterval(projectPollTimer);
    projectPollTimer = setInterval(() => {
      if (document.visibilityState === "visible" && !$('taskDialog')?.open) {
        loadProjects(projectPage, { silent: true }).catch(() => {});
      }
    }, 5000);
  } catch (error) {
    toast(`加载失败：${error.message}`);
  }
}

$("projectSearch").addEventListener("input", event => {
  projectSearchTerm = String(event.target.value || "");
  projectPage = 1;
  clearTimeout(projectSearchTimer);
  projectSearchTimer = setTimeout(() => loadProjects(1).catch(error => toast(`搜索失败：${error.message}`)), 220);
});

$("projectPrev").addEventListener("click", () => {
  if (projectPage <= 1) return;
  projectPage -= 1;
  loadProjects(projectPage).catch(error => toast(`加载失败：${error.message}`));
});

$("projectNext").addEventListener("click", () => {
  if (projectPage >= projectTotalPages) return;
  projectPage += 1;
  loadProjects(projectPage).catch(error => toast(`加载失败：${error.message}`));
});

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    loadProjects(projectPage, { silent: true }).catch(() => {});
    if (!reviewPlatformMode) loadTasks({ silent: true });
  }
});

async function loadTasks({ silent = false } = {}) {
  if (taskPollInFlight) return;
  taskPollInFlight = true;
  try {
    const tasks = await api("/api/project/tasks");
    tasks.forEach(task => {
      const id = String(task.task_id || "");
      const previous = taskStatusHistory.get(id);
      const current = String(task.status || "queued");
      if (previous && previous !== current && current !== "queued") {
        toast(`${task.name || "任务"}：${taskStatusInfo(task).label}`);
      }
      taskStatusHistory.set(id, current);
    });
    const queueTasks = tasks.filter(task => String(task.status) !== "completed");
    $("taskRows").innerHTML = queueTasks.length
      ? queueTasks.slice().reverse().map(renderTask).join("")
      : `<div class="empty">暂无待执行任务</div>`;
    document.querySelectorAll("[data-task-action]").forEach(button => {
      button.addEventListener("click", handleTaskAction);
    });
    updateTaskPolling(queueTasks);
  } catch (error) {
    if (!silent) throw error;
  } finally {
    taskPollInFlight = false;
  }
}

function updateTaskPolling(tasks) {
  if (taskPollTimer) clearInterval(taskPollTimer);
  taskPollTimer = null;
  const shouldPoll = Array.isArray(tasks) && tasks.some(task =>
    ["queued", "running"].includes(String(task.status || "queued"))
  );
  if (!shouldPoll) return;
  taskPollTimer = setInterval(() => {
    if (document.visibilityState === "hidden" || $("taskDialog")?.open) return;
    loadTasks({ silent: true });
  }, 2500);
}

function renderAnalysis(value) {
  analysis = value;
  applyDefaultTaskName(value.folder_name || folderNameFromPath($("folderPath").value));
  const options = value.timepoint_options || [];
  $("analysisResult").textContent = `已解析：${value.group_count} 个板组，${value.session_count} 个时间点文件夹\n实际日龄：${(value.day_labels || []).join(", ")}\n完整96孔组：${value.complete_groups}`;
  const picker = $("timepointPicker");
  picker.hidden = false;
  const defaults = new Set(value.default_selected_timepoint_labels || []);
  picker.innerHTML = options.map(item => {
    const required = item.required_early;
    const checked = defaults.has(item.day_label) ? "checked" : "";
    return `<label class="timepoint-choice ${required ? "required" : ""}"><input type="checkbox" data-day="${esc(item.day_label)}" ${checked}><span>${esc(item.day_label)}${required ? " · 必选" : ""}</span><small>${item.group_count || 0}组</small></label>`;
  }).join("");
  $("endpointNote").hidden = false;
  updateEndpoint();
  picker.querySelectorAll("input").forEach(input => input.addEventListener("change", updateEndpoint));
  $("queueButton").disabled = !value.selection_valid;
}

function selectedDays() {
  return [...$("timepointPicker").querySelectorAll("input:checked")].map(input => input.dataset.day);
}

function updateEndpoint() {
  const selected = selectedDays();
  const options = analysis?.timepoint_options || [];
  const later = options
    .filter(item => selected.includes(item.day_label) && count(item.day_number) >= 7)
    .sort((left, right) => count(left.day_number) - count(right.day_number));
  const missing = ["Day0", "Day1", "Day2"].filter(day => !selected.includes(day));
  const valid = !missing.length && later.length;
  $("endpointNote").textContent = valid
    ? `末点：${later[later.length - 1].day_label}（默认使用最后勾选的后期时间点做生长快速排除）`
    : `当前不能加入队列：${missing.length ? `缺少 ${missing.join("、")}；` : "至少选择一个 Day7 或更晚时间点；"}`;
  $("queueButton").disabled = !valid;
}

function closeTaskDialog() {
  const dialog = $("taskDialog");
  if (dialog?.open) dialog.close("cancel");
}

function openNewTaskDialog() {
  $("taskDialog").showModal();
  analysis = null;
  $("taskName").value = "";
  $("taskName").dataset.edited = "";
  taskNameAutoValue = "";
  $("createdBy").value = "";
  $("queueButton").disabled = true;
  $("timepointPicker").hidden = true;
  $("endpointNote").hidden = true;
  $("analysisResult").textContent = "选择文件夹后点击解析。";
}

function folderNameFromPath(value) {
  const parts = String(value || "").replace(/[\\/]+$/, "").split(/[\\/]/).filter(Boolean);
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

$("refreshButton").addEventListener("click", load);
$("closeTaskButton").addEventListener("click", closeTaskDialog);
$("cancelTaskButton").addEventListener("click", closeTaskDialog);
$("newTaskButton").addEventListener("click", async () => {
  if (!reviewPlatformMode) {
    openNewTaskDialog();
    return;
  }
  const button = $("newTaskButton");
  button.disabled = true;
  try {
    toast("正在打开审核数据选择窗口…");
    const result = await api("/api/review-platform/import-package", { method: "POST" });
    reviewPlatformStatus = result;
    if (result.status === "cancelled") {
      toast("未选择审核数据");
      return;
    }
    toast(`已导入：${result.project_name || result.project_id}`);
    projectPage = 1;
    await loadProjects(1);
  } catch (error) {
    toast(`导入失败：${error.message}`);
  } finally {
    button.disabled = false;
  }
});
$("browseButton").addEventListener("click", async () => {
  $("analysisResult").textContent = "正在打开文件夹选择器…";
  try {
    const result = await api("/api/project/browse-folder", { method: "POST" });
    if (result.path) {
      $("folderPath").value = result.path;
      // Selecting a folder is the user's explicit confirmation; start the
      // same analysis as the manual button immediately so the available days,
      // board count and endpoint are visible without a second click.
      await analyzeFolder();
    } else {
      $("analysisResult").textContent = result.error || "未选择文件夹；也可以直接输入文件夹路径。";
    }
  } catch (error) {
    $("analysisResult").textContent = `解析失败：${error.message}`;
    $("queueButton").disabled = true;
  }
});

async function analyzeFolder() {
  try {
    renderAnalysis(await api("/api/project/analyze-folder", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: $("folderPath").value.trim() }),
    }));
  } catch (error) {
    $("analysisResult").textContent = `解析失败：${error.message}`;
    $("queueButton").disabled = true;
  }
}

$("analyzeButton").addEventListener("click", analyzeFolder);
$("taskName").addEventListener("input", () => { $("taskName").dataset.edited = "1"; });
$("queueButton").addEventListener("click", async () => {
  try {
    const task = await api("/api/project/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: $("taskName").value.trim(),
        created_by: $("createdBy").value.trim(),
        path: $("folderPath").value.trim(),
        selected_timepoint_labels: selectedDays(),
      }),
    });
    $("taskDialog").close();
    toast(`已加入任务队列：${task.name}，末点 ${task.endpoint_day_label}`);
    await loadTasks();
  } catch (error) {
    toast(`加入队列失败：${error.message}`);
  }
});

load();
