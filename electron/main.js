"use strict";

const { app, BrowserWindow, Menu, dialog, ipcMain, shell } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");
const {
  commandLineDeploymentRoot,
  configuredPort,
  findDevelopmentDeploymentRoot,
  isAllowedClientUrl,
  normalizeWorkspaceSelection,
} = require("./lib/runtime");

let mainWindow = null;
let startupPromise = null;
let deploymentRoot = "";
let productionPort = 8777;
let lastStartupStatus = { message: "正在初始化客户端……", kind: "progress" };
const bootstrapLog = path.join(process.env.LOCALAPPDATA || process.env.TEMP || __dirname, "CellVision", "electron-bootstrap.log");

function logBootstrap(message) {
  try {
    fs.mkdirSync(path.dirname(bootstrapLog), { recursive: true });
    fs.appendFileSync(bootstrapLog, `[${new Date().toISOString()}] ${message}\n`, "utf8");
  } catch {}
}

logBootstrap(`process started; packaged=${app.isPackaged}; executable=${process.execPath}`);
process.on("uncaughtException", (error) => logBootstrap(`uncaughtException: ${error.stack || error}`));
process.on("unhandledRejection", (error) => logBootstrap(`unhandledRejection: ${error?.stack || error}`));

function resolveDeploymentRoot() {
  const commandLine = commandLineDeploymentRoot(process.argv);
  if (commandLine) return commandLine;
  const explicit = process.env.CELLVISION_DEPLOYMENT_ROOT;
  if (explicit) return path.resolve(explicit);
  if (app.isPackaged) return path.dirname(process.execPath);
  return findDevelopmentDeploymentRoot(path.resolve(__dirname, ".."));
}

function assertDeployment() {
  const required = [
    path.join(deploymentRoot, "Application", ".env.production"),
    path.join(deploymentRoot, "start_cellvision_platform.ps1"),
    path.join(deploymentRoot, "recover_cellvision_projects_launcher.ps1"),
    path.join(deploymentRoot, "import_cellvision_workspace_launcher.ps1"),
  ];
  for (const requiredPath of required) {
    if (!fs.existsSync(requiredPath)) {
      throw new Error(`安装目录不完整，缺少：${requiredPath}`);
    }
  }
}

function sendStatus(message, kind = "progress") {
  lastStartupStatus = { message, kind };
  logBootstrap(`${kind}: ${message}`);
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("cellvision:startup-status", lastStartupStatus);
  }
}

function runPowerShell(scriptName, args = [], timeoutMs = 10 * 60 * 1000) {
  const scriptPath = path.join(deploymentRoot, scriptName);
  return new Promise((resolve, reject) => {
    const child = spawn(
      "powershell.exe",
      ["-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", scriptPath, ...args],
      { cwd: deploymentRoot, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] },
    );
    let stdout = "";
    let stderr = "";
    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    const timer = setTimeout(() => {
      child.kill();
      reject(new Error(`${scriptName} 执行超时。`));
    }, timeoutMs);
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code === 0) resolve({ stdout, stderr });
      else reject(new Error((stderr || stdout || `${scriptName} 执行失败（${code}）`).trim()));
    });
  });
}

async function apiJson(route, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`http://127.0.0.1:${productionPort}/${route.replace(/^\//, "")}`, {
      signal: controller.signal,
      cache: "no-store",
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}

async function ensurePlatform() {
  if (startupPromise) return startupPromise;
  startupPromise = (async () => {
    sendStatus("正在检查 Cell Vision 服务和计算工作器……");
    await runPowerShell("start_cellvision_platform.ps1", ["-NoOpen", "-StartupTimeoutSeconds", "180"]);
    const ready = await apiJson("api/ready");
    if (ready.status !== "ready") throw new Error("Cell Vision 服务尚未就绪。");
    sendStatus("服务已就绪，正在打开项目主页……", "success");
    await mainWindow.loadURL(`http://127.0.0.1:${productionPort}/?electron=1`);
  })();
  try {
    await startupPromise;
  } catch (error) {
    sendStatus(error.message || String(error), "error");
    throw error;
  } finally {
    startupPromise = null;
  }
}

function authorizedSender(event) {
  const value = event.senderFrame?.url || event.sender.getURL();
  if (!isAllowedClientUrl(value, productionPort)) throw new Error("拒绝非本机页面调用桌面功能。");
}

async function repairProjects() {
  sendStatus("正在备份并修复项目目录……");
  const result = await runPowerShell("recover_cellvision_projects_launcher.ps1", ["-StartupTimeoutSeconds", "180"]);
  await apiJson("api/catalog/status");
  if (mainWindow && !mainWindow.isDestroyed()) await mainWindow.reload();
  return { ok: true, message: "项目目录已修复并重新扫描。", output: result.stdout };
}

async function importWorkspace() {
  const selection = await dialog.showOpenDialog(mainWindow, {
    title: "选择旧的 Cell Vision Workspace",
    properties: ["openDirectory"],
    buttonLabel: "导入此 Workspace",
  });
  if (selection.canceled || !selection.filePaths.length) return { ok: false, canceled: true };
  const sourceWorkspace = normalizeWorkspaceSelection(selection.filePaths[0]);
  if (!fs.existsSync(path.join(sourceWorkspace, "Projects"))) {
    throw new Error("所选目录不是有效的 Workspace：未找到 Projects 文件夹。");
  }
  const confirmation = await dialog.showMessageBox(mainWindow, {
    type: "question",
    title: "导入历史 Workspace",
    message: "导入时会暂时停止计算服务，并先备份当前项目元数据。",
    detail: `来源：${sourceWorkspace}\n目标：${path.join(deploymentRoot, "Workspace")}\n\n已有同名项目不会被覆盖。是否继续？`,
    buttons: ["开始导入", "取消"],
    defaultId: 0,
    cancelId: 1,
  });
  if (confirmation.response !== 0) return { ok: false, canceled: true };
  sendStatus("正在导入历史 Workspace；数据量较大时可能需要较长时间……");
  await runPowerShell("import_cellvision_workspace_launcher.ps1", ["-SourceWorkspace", sourceWorkspace], 24 * 60 * 60 * 1000);
  await apiJson("api/catalog/status");
  if (mainWindow && !mainWindow.isDestroyed()) await mainWindow.loadURL(`http://127.0.0.1:${productionPort}/?electron=1`);
  return { ok: true, message: "历史 Workspace 已导入并重新扫描。" };
}

function installIpcHandlers() {
  ipcMain.handle("cellvision:choose-folder", async (event) => {
    authorizedSender(event);
    const result = await dialog.showOpenDialog(mainWindow, { properties: ["openDirectory"] });
    return { path: result.canceled ? "" : result.filePaths[0] || "", error: "" };
  });
  ipcMain.handle("cellvision:retry-startup", async (event) => {
    authorizedSender(event);
    await ensurePlatform();
    return { ok: true };
  });
  ipcMain.handle("cellvision:repair-projects", async (event) => {
    authorizedSender(event);
    return repairProjects();
  });
  ipcMain.handle("cellvision:import-workspace", async (event) => {
    authorizedSender(event);
    return importWorkspace();
  });
  ipcMain.handle("cellvision:open-logs", async (event) => {
    authorizedSender(event);
    const logs = path.join(deploymentRoot, "Workspace", "Logs");
    fs.mkdirSync(logs, { recursive: true });
    await shell.openPath(logs);
    return { ok: true };
  });
}

function installMenu() {
  const template = [
    {
      label: "平台",
      submenu: [
        { label: "刷新页面", accelerator: "F5", click: () => mainWindow?.reload() },
        { label: "返回项目主页", click: () => mainWindow?.loadURL(`http://127.0.0.1:${productionPort}/?electron=1`) },
        { type: "separator" },
        { label: "退出", accelerator: "Alt+F4", click: () => app.quit() },
      ],
    },
    {
      label: "维护",
      submenu: [
        {
          label: "修复/重新扫描项目目录",
          click: async () => {
            try {
              await repairProjects();
              await dialog.showMessageBox(mainWindow, { type: "info", message: "项目目录修复完成。" });
            } catch (error) {
              await dialog.showMessageBox(mainWindow, { type: "error", message: "项目目录修复失败", detail: error.message });
            }
          },
        },
        {
          label: "导入历史 Workspace",
          click: async () => {
            try {
              const result = await importWorkspace();
              if (result.ok) await dialog.showMessageBox(mainWindow, { type: "info", message: result.message });
            } catch (error) {
              await dialog.showMessageBox(mainWindow, { type: "error", message: "Workspace 导入失败", detail: error.message });
            }
          },
        },
        { label: "打开日志目录", click: () => shell.openPath(path.join(deploymentRoot, "Workspace", "Logs")) },
      ],
    },
    { label: "帮助", submenu: [{ label: "关于 Cell Vision", role: "about" }] },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1500,
    height: 960,
    minWidth: 1100,
    minHeight: 720,
    show: false,
    backgroundColor: "#f4f1e8",
    title: "Cell Vision",
    autoHideMenuBar: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (isAllowedClientUrl(url, productionPort)) return { action: "allow" };
    shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (!isAllowedClientUrl(url, productionPort)) event.preventDefault();
  });
  mainWindow.once("ready-to-show", () => mainWindow.show());
  mainWindow.webContents.on("did-finish-load", () => {
    if (mainWindow?.webContents.getURL().startsWith("file:")) {
      mainWindow.webContents.send("cellvision:startup-status", lastStartupStatus);
    }
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  mainWindow.loadFile(path.join(__dirname, "splash.html"));
}

const singleInstance = app.requestSingleInstanceLock();
if (!singleInstance) {
  logBootstrap("another Cell Vision client instance owns the single-instance lock");
  app.quit();
}
else {
  app.on("second-instance", () => {
    if (mainWindow) {
      if (mainWindow.isMinimized()) mainWindow.restore();
      mainWindow.focus();
    }
  });
  app.whenReady().then(async () => {
    app.setAppUserModelId("com.cellvision.production.desktop");
    deploymentRoot = resolveDeploymentRoot();
    logBootstrap(`deployment root=${deploymentRoot}`);
    if (!deploymentRoot) {
      dialog.showErrorBox("Cell Vision 安装不完整", "未找到生产部署目录。开发模式请设置 CELLVISION_DEPLOYMENT_ROOT。");
      app.quit();
      return;
    }
    try {
      assertDeployment();
      productionPort = configuredPort(deploymentRoot);
    } catch (error) {
      dialog.showErrorBox("Cell Vision 安装不完整", error.message);
      app.quit();
      return;
    }
    installIpcHandlers();
    installMenu();
    createWindow();
    logBootstrap("browser window created");
    try {
      await ensurePlatform();
    } catch {}
  });
}

app.on("window-all-closed", () => app.quit());
