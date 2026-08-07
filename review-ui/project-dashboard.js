const $ = id => document.getElementById(id);
let analysis = null;
const projectId = document.querySelector('meta[name="project-id"]')?.content || "";
const names = { no_obvious_growth: "无明显生长", single_cell_origin: "单细胞来源", multi_cell_origin: "多细胞来源", undetermined: "待确定", positive_control: "阳性对照" };
const toast = text => { $("toast").textContent = text; $("toast").classList.add("show"); setTimeout(() => $("toast").classList.remove("show"), 2400); };
const esc = value => String(value ?? "").replace(/[&<>\"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
async function api(url, options) { const response = await fetch(url, options); if (!response.ok) throw new Error(await response.text()); return response.json(); }
function summaryCard(label, value) { return `<div class="summary-card"><span>${label}</span><strong>${Number(value || 0).toLocaleString()}</strong></div>`; }
function renderProject(data) {
  $("projectName").textContent = data.project_name || data.project_id;
  $("projectMeta").textContent = `${data.plate_count} 块板 · ${data.generated_at ? new Date(data.generated_at).toLocaleString() : "项目级汇总"}`;
  $("mountedMeta").textContent = `${data.mounted_plates.length} / ${data.plate_count} 块可审核`;
  const counts = data.category_counts || {};
  $("summaryCards").innerHTML = [summaryCard("无明显生长", counts.no_obvious_growth), summaryCard("单细胞来源", counts.single_cell_origin), summaryCard("多细胞来源", counts.multi_cell_origin), summaryCard("待确定", counts.undetermined), summaryCard("阳性对照", counts.positive_control)].join("");
  $("plateRows").innerHTML = data.plates.length ? data.plates.map(plate => {
    const ready = data.mounted_plates.includes(plate.slug); const target = ready ? `/plates/${encodeURIComponent(plate.slug)}/` : "#"; const status = plate.status || "queued";
    const counts = plate.category_counts || {};
    return `<tr class="${ready ? "clickable" : ""}" ${ready ? `data-href="${target}"` : ""}><td><span class="plate-link">${esc(plate.board_id || plate.group_id)}</span><br><small>${esc(plate.group_id)}</small></td><td><span class="status ${status}">${status === "completed" ? "已完成" : status === "running" ? "计算中" : status === "error" ? "失败" : "排队"}</span></td><td>${counts.no_obvious_growth || 0}</td><td>${counts.single_cell_origin || 0}</td><td>${counts.multi_cell_origin || 0}</td><td>${counts.undetermined || 0}</td><td>${plate.well_count || 0}</td><td>${plate.elapsed_seconds ? `${Number(plate.elapsed_seconds).toFixed(1)} s` : "—"}</td><td>${ready ? "进入审核 →" : ""}</td></tr>`;
  }).join("") : `<tr><td colspan="9" class="empty">暂无板子</td></tr>`;
  document.querySelectorAll("tr[data-href]").forEach(row => row.addEventListener("click", () => { window.location.href = row.dataset.href; }));
}
async function loadProject() { try { renderProject(await api(`/api/project${projectId ? `?project_id=${encodeURIComponent(projectId)}` : ""}`)); await loadTasks(); } catch (error) { toast(`加载失败：${error.message}`); } }
async function loadTasks() { const tasks = await api("/api/project/tasks"); $("taskRows").innerHTML = tasks.length ? tasks.slice().reverse().map(task => `<div class="task"><strong>${esc(task.name)}</strong><span>${esc(task.path)}</span><span class="status">${esc(task.status)}</span><small>${task.group_count || 0} 组 · ${new Date(task.created_at).toLocaleString()}</small></div>`).join("") : `<div class="empty">暂无任务</div>`; }
function showAnalysis(value) { analysis = value; $("analysisResult").textContent = `已解析：${value.group_count} 个板组，${value.session_count} 个时间点文件夹\n时间点：${value.timepoint_labels.join(", ")} · 日龄：${value.day_labels.join(", ")}\n完整96孔组：${value.complete_groups}`; $("queueButton").disabled = false; }
$("refreshButton").addEventListener("click", loadProject);
$("browseButton").addEventListener("click", async () => { const result = await api("/api/project/browse-folder", { method: "POST" }); if (result.path) $("folderPath").value = result.path; });
$("analyzeButton").addEventListener("click", async () => { try { showAnalysis(await api("/api/project/analyze-folder", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path: $("folderPath").value.trim() }) })); } catch (error) { $("analysisResult").textContent = `解析失败：${error.message}`; $("queueButton").disabled = true; } });
$("queueButton").addEventListener("click", async () => { try { const task = await api("/api/project/tasks", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: $("taskName").value.trim(), path: $("folderPath").value.trim() }) }); $("taskDialog").close(); toast(`已加入任务队列：${task.name}`); await loadTasks(); } catch (error) { toast(`加入队列失败：${error.message}`); } });
loadProject();
