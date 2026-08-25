"use strict";

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("cellvisionDesktop", {
  chooseFolder: () => ipcRenderer.invoke("cellvision:choose-folder"),
  importWorkspace: () => ipcRenderer.invoke("cellvision:import-workspace"),
  repairProjects: () => ipcRenderer.invoke("cellvision:repair-projects"),
  retryStartup: () => ipcRenderer.invoke("cellvision:retry-startup"),
  openLogs: () => ipcRenderer.invoke("cellvision:open-logs"),
  onStartupStatus: (listener) => {
    const wrapped = (_event, value) => listener(value);
    ipcRenderer.on("cellvision:startup-status", wrapped);
    return () => ipcRenderer.removeListener("cellvision:startup-status", wrapped);
  },
});
