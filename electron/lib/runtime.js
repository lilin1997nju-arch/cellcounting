"use strict";

const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

function parseEnvironment(text) {
  const values = {};
  for (const rawLine of String(text || "").split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const separator = line.indexOf("=");
    if (separator <= 0) continue;
    values[line.slice(0, separator).trim()] = line.slice(separator + 1).trim();
  }
  return values;
}

function configuredPort(deploymentRoot, explicitPort = 0) {
  if (Number(explicitPort) > 0) return Number(explicitPort);
  const environmentPath = path.join(deploymentRoot, "Application", ".env.production");
  try {
    const values = parseEnvironment(fs.readFileSync(environmentPath, "utf8"));
    const port = Number(values.CELLVISION_PORT);
    if (Number.isInteger(port) && port > 0 && port < 65536) return port;
  } catch {}
  return 8777;
}

function ensureInstanceId(deploymentRoot) {
  const workspaceRoot = path.join(deploymentRoot, "Workspace");
  const instancePath = path.join(workspaceRoot, ".cellvision-instance-id");
  fs.mkdirSync(workspaceRoot, { recursive: true });
  try {
    const current = fs.readFileSync(instancePath, "utf8").trim();
    if (current) return current;
  } catch {}
  const value = crypto.randomUUID();
  try {
    fs.writeFileSync(instancePath, `${value}\n`, { encoding: "utf8", flag: "wx" });
    return value;
  } catch (error) {
    if (error?.code !== "EEXIST") throw error;
    const current = fs.readFileSync(instancePath, "utf8").trim();
    if (!current) throw new Error(`Cell Vision 实例标识为空：${instancePath}`);
    return current;
  }
}

function normalizeWorkspaceSelection(selectedPath) {
  if (!selectedPath) return "";
  const selected = path.resolve(selectedPath);
  if (fs.existsSync(path.join(selected, "Projects"))) return selected;
  const nested = path.join(selected, "Workspace");
  if (fs.existsSync(path.join(nested, "Projects"))) return nested;
  return selected;
}

function isAllowedClientUrl(value, port) {
  try {
    const url = new URL(value);
    if (url.protocol === "file:") return true;
    return (
      url.protocol === "http:" &&
      ["127.0.0.1", "localhost"].includes(url.hostname) &&
      Number(url.port || 80) === Number(port)
    );
  } catch {
    return false;
  }
}

function findDevelopmentDeploymentRoot(repositoryRoot) {
  const releaseRoot = path.join(repositoryRoot, "release");
  if (!fs.existsSync(releaseRoot)) return "";
  const candidates = fs
    .readdirSync(releaseRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(releaseRoot, entry.name))
    .filter((candidate) => fs.existsSync(path.join(candidate, "Application", ".env.production")))
    .sort((left, right) => fs.statSync(right).mtimeMs - fs.statSync(left).mtimeMs);
  return candidates[0] || "";
}

function commandLineDeploymentRoot(argv) {
  const prefix = "--deployment-root=";
  const value = (argv || []).find((argument) => String(argument).startsWith(prefix));
  return value ? path.resolve(String(value).slice(prefix.length)) : "";
}

module.exports = {
  commandLineDeploymentRoot,
  configuredPort,
  ensureInstanceId,
  findDevelopmentDeploymentRoot,
  isAllowedClientUrl,
  normalizeWorkspaceSelection,
  parseEnvironment,
};
