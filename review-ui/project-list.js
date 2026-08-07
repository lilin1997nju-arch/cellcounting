const $ = id => document.getElementById(id);
let analysis = null;

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

function count(value) {
  return Number(value || 0);
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

function projectStatus(item) {
  const total = count(item.plate_count);
  const recognized = count(item.recognized_plate_count ?? item.completed_plate_count);
  const reviewed = count(item.reviewed_plate_count);
  if (total > 0 && reviewed >= total) return { label: "已审核", className: "reviewed" };
  if (total > 0 && recognized >= total) return { label: "已完成识别", className: "recognized" };
  if (recognized > 0) return { label: "识别中", className: "processing" };
  return { label: "待识别", className: "pending" };
}

function renderProjects(items) {
  const rows = $("projectRows");
  $("projectCount").textContent = `${items.length} 个项目`;

  let boards = 0;
  let recognized = 0;
  items.forEach(item => {
    boards += count(item.plate_count);
    recognized += count(item.recognized_plate_count ?? item.completed_plate_count);
  });

  $("summaryCards").innerHTML = [
    summaryCard("项目", items.length),
    summaryCard("板子", boards),
    summaryCard("已完成识别板子", recognized),
    summaryCard("待处理板子", Math.max(0, boards - recognized)),
  ].join("");

  if (!items.length) {
    rows.innerHTML = `<tr><td colspan="6" class="empty">暂无项目，请新建任务。</td></tr>`;
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
    return `<tr class="project-row" data-url="${esc(item.detail_url)}">
      <td><strong class="project-name">${esc(item.project_name)}</strong><span class="project-subline">${esc(item.root || "未记录数据根目录")}</span></td>
      <td>${dateRange(item)}</td>
      <td>${esc(item.created_by || "—")}</td>
      <td><span class="project-status ${status.className}">${status.label}</span><span class="project-status-detail">已完成识别 ${recognizedCount}/${total} · 已审核 ${reviewedCount}/${total}</span></td>
      <td><span class="project-result">单细胞来源孔：${singleCell.toLocaleString()}</span><span class="project-subline">多细胞来源 ${multiCell.toLocaleString()} · 待确定 ${undetermined.toLocaleString()}</span></td>
      <td><a class="project-open" href="${esc(item.detail_url)}">进入项目 →</a></td>
    </tr>`;
  }).join("");

  rows.querySelectorAll("tr[data-url]").forEach(row => row.addEventListener("click", event => {
    if (event.target.closest("a")) return;
    window.location.href = row.dataset.url;
  }));
}

async function load() {
  try {
    renderProjects(await api("/api/projects"));
    await loadTasks();
  } catch (error) {
    toast(`加载失败：${error.message}`);
  }
}

async function loadTasks() {
  const tasks = await api("/api/project/tasks");
  $("taskRows").innerHTML = tasks.length
    ? tasks.slice().reverse().map(task => `<div class="task"><strong>${esc(task.name)}</strong><span>${esc(task.path)}</span><span class="status ${esc(task.status)}">${esc(task.status)}</span><small>${task.group_count || 0} 组 · 已选 ${esc((task.selected_timepoint_labels || []).join(", "))} · 末点 ${esc(task.endpoint_day_label || "—")}</small></div>`).join("")
    : `<div class="empty">暂无任务</div>`;
}

function renderAnalysis(value) {
  analysis = value;
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

$("refreshButton").addEventListener("click", load);
$("closeTaskButton").addEventListener("click", closeTaskDialog);
$("cancelTaskButton").addEventListener("click", closeTaskDialog);
$("newTaskButton").addEventListener("click", () => {
  $("taskDialog").showModal();
  analysis = null;
  $("queueButton").disabled = true;
  $("timepointPicker").hidden = true;
  $("endpointNote").hidden = true;
  $("analysisResult").textContent = "选择文件夹后点击解析。";
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
$("queueButton").addEventListener("click", async () => {
  try {
    const task = await api("/api/project/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: $("taskName").value.trim(),
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
