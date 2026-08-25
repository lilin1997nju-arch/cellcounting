"use strict";

const status = document.getElementById("status");
const actions = document.getElementById("actions");
const retry = document.getElementById("retry");
const logs = document.getElementById("logs");

window.cellvisionDesktop.onStartupStatus((value) => {
  status.textContent = value.message;
  status.dataset.kind = value.kind;
  actions.hidden = value.kind !== "error";
});

retry.addEventListener("click", async () => {
  actions.hidden = true;
  status.textContent = "正在重新检查服务……";
  try {
    await window.cellvisionDesktop.retryStartup();
  } catch (error) {
    status.textContent = error.message || String(error);
    status.dataset.kind = "error";
    actions.hidden = false;
  }
});

logs.addEventListener("click", () => window.cellvisionDesktop.openLogs());
