#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process";

const [, , toolName, rawArgs = "{}"] = process.argv;

function usage() {
  console.error("Usage: mcp-tool <tool-name|--list> [json-args]");
  console.error("Set CONTEXTBENCH_MCP_COMMAND and optional CONTEXTBENCH_MCP_ARGS_JSON.");
  process.exit(2);
}

if (!toolName) {
  usage();
}

function parseJsonObject(value, label) {
  try {
    const parsed = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      throw new Error("value must be a JSON object");
    }
    return parsed;
  } catch (error) {
    console.error(`mcp-tool: invalid ${label}: ${error.message}`);
    process.exit(2);
  }
}

function parseJsonArray(value, label) {
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed) || !parsed.every((item) => typeof item === "string")) {
      throw new Error("value must be a JSON string array");
    }
    return parsed;
  } catch (error) {
    console.error(`mcp-tool: invalid ${label}: ${error.message}`);
    process.exit(2);
  }
}

const toolArgs = toolName === "--list" ? {} : parseJsonObject(rawArgs, "json-args");
const command = (process.env.CONTEXTBENCH_MCP_COMMAND || "").trim();
if (!command) {
  console.error("mcp-tool: CONTEXTBENCH_MCP_COMMAND is required");
  process.exit(2);
}

const commandArgs = parseJsonArray(process.env.CONTEXTBENCH_MCP_ARGS_JSON || "[]", "CONTEXTBENCH_MCP_ARGS_JSON");
const cwd = (process.env.CONTEXTBENCH_MCP_CWD || "").trim() || undefined;
const clientName = (process.env.CONTEXTBENCH_MCP_CLIENT_NAME || "agentstackbench-mcp-tool").trim();
const timeoutMs = Number.parseInt(process.env.CONTEXTBENCH_MCP_TIMEOUT_MS || "30000", 10);
const child = spawn(command, commandArgs, { cwd, env: process.env, stdio: ["pipe", "pipe", "pipe"] });

let stderr = "";
let buffer = "";
let completed = false;

const timer = setTimeout(() => {
  if (!completed) {
    child.kill("SIGTERM");
    console.error(`mcp-tool: timed out after ${timeoutMs}ms`);
    if (stderr.trim()) {
      console.error(stderr.trim());
    }
    process.exit(124);
  }
}, Number.isFinite(timeoutMs) && timeoutMs > 0 ? timeoutMs : 30000);

function finish(code) {
  completed = true;
  clearTimeout(timer);
  try {
    child.kill("SIGTERM");
  } catch {
    // Process may already be gone.
  }
  process.exit(code);
}

function handleMessage(message) {
  if (message.id !== 2) {
    return;
  }
  if (message.error) {
    console.error(JSON.stringify(message.error, null, 2));
    finish(1);
    return;
  }
  process.stdout.write(`${JSON.stringify(message.result, null, 2)}\n`);
  finish(0);
}

child.stderr.on("data", (chunk) => {
  stderr += chunk.toString("utf8");
});

child.stdout.on("data", (chunk) => {
  buffer += chunk.toString("utf8");
  let newline = buffer.indexOf("\n");
  while (newline !== -1) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (line) {
      try {
        handleMessage(JSON.parse(line));
      } catch (error) {
        console.error(`mcp-tool: invalid JSON-RPC response: ${error.message}`);
        console.error(line);
        finish(1);
      }
    }
    newline = buffer.indexOf("\n");
  }
});

child.on("error", (error) => {
  console.error(`mcp-tool: failed to start ${command}: ${error.message}`);
  finish(127);
});

child.on("exit", (code, signal) => {
  if (completed) {
    return;
  }
  if (buffer.trim()) {
    try {
      handleMessage(JSON.parse(buffer.trim()));
      return;
    } catch {
      // Fall through to the process-exit error below.
    }
  }
  const detail = stderr.trim();
  if (detail) {
    console.error(detail);
  }
  console.error(`mcp-tool: MCP server exited before response (${signal || code})`);
  finish(code || 1);
});

const init = {
  jsonrpc: "2.0",
  id: 1,
  method: "initialize",
  params: {
    protocolVersion: "2025-03-26",
    capabilities: {},
    clientInfo: { name: clientName, version: "0.1.0" },
  },
};
const initialized = {
  jsonrpc: "2.0",
  method: "notifications/initialized",
  params: {},
};
const request =
  toolName === "--list"
    ? { jsonrpc: "2.0", id: 2, method: "tools/list", params: {} }
    : {
        jsonrpc: "2.0",
        id: 2,
        method: "tools/call",
        params: { name: toolName, arguments: toolArgs },
      };

for (const message of [init, initialized, request]) {
  child.stdin.write(`${JSON.stringify(message)}\n`);
}
