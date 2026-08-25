"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const {
  commandLineDeploymentRoot,
  configuredPort,
  isAllowedClientUrl,
  normalizeWorkspaceSelection,
  parseEnvironment,
} = require("../lib/runtime");

test("production environment parser keeps values after the first separator", () => {
  assert.deepEqual(parseEnvironment("# comment\nCELLVISION_PORT=8778\nVALUE=a=b\n"), {
    CELLVISION_PORT: "8778",
    VALUE: "a=b",
  });
});

test("configured port reads the bundled production environment", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "cellvision-electron-"));
  fs.mkdirSync(path.join(root, "Application"));
  fs.writeFileSync(path.join(root, "Application", ".env.production"), "CELLVISION_PORT=8899\n");
  assert.equal(configuredPort(root), 8899);
  assert.equal(configuredPort(root, 9001), 9001);
});

test("workspace selector accepts either Workspace or its parent", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "cellvision-workspace-"));
  fs.mkdirSync(path.join(root, "Workspace", "Projects"), { recursive: true });
  assert.equal(normalizeWorkspaceSelection(root), path.join(root, "Workspace"));
  assert.equal(normalizeWorkspaceSelection(path.join(root, "Workspace")), path.join(root, "Workspace"));
});

test("desktop privileges are limited to local production and splash pages", () => {
  assert.equal(isAllowedClientUrl("file:///splash.html", 8777), true);
  assert.equal(isAllowedClientUrl("http://127.0.0.1:8777/projects/demo", 8777), true);
  assert.equal(isAllowedClientUrl("http://localhost:8777/", 8777), true);
  assert.equal(isAllowedClientUrl("https://example.com/", 8777), false);
  assert.equal(isAllowedClientUrl("http://127.0.0.1:9999/", 8777), false);
});

test("diagnostic deployment root can be supplied on the command line", () => {
  assert.equal(
    commandLineDeploymentRoot(["Cell Vision.exe", "--deployment-root=C:\\CellVision"]),
    path.resolve("C:\\CellVision"),
  );
  assert.equal(commandLineDeploymentRoot(["Cell Vision.exe"]), "");
});
