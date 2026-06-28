#!/usr/bin/env node
// SPDX-License-Identifier: Apache-2.0

import http from "node:http";

function fail(message) {
  console.error(message);
  process.exit(2);
}

function parseRules() {
  const raw = process.env.CONTEXTBENCH_PROXY_REWRITE_RULES_JSON || "[]";
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch (error) {
    fail(`Invalid CONTEXTBENCH_PROXY_REWRITE_RULES_JSON: ${error.message}`);
  }
  if (!Array.isArray(parsed)) {
    fail("CONTEXTBENCH_PROXY_REWRITE_RULES_JSON must be a JSON array");
  }
  return parsed.map((rule, index) => {
    if (!rule || typeof rule !== "object" || Array.isArray(rule)) {
      fail(`Proxy rewrite rule ${index} must be a JSON object`);
    }
    const path = typeof rule.path === "string" && rule.path ? rule.path : null;
    if (!path) {
      fail(`Proxy rewrite rule ${index} requires a non-empty path`);
    }
    const methods = Array.isArray(rule.methods) && rule.methods.length ? rule.methods : ["POST", "PUT", "PATCH"];
    const normalizedMethods = methods.map((method) => String(method).toUpperCase());
    const set = rule.set;
    if (!set || typeof set !== "object" || Array.isArray(set)) {
      fail(`Proxy rewrite rule ${index} requires a set object`);
    }
    for (const [field, value] of Object.entries(set)) {
      if (!field || !value || typeof value !== "object" || Array.isArray(value)) {
        fail(`Proxy rewrite rule ${index} field ${field} must be an object`);
      }
      if ("env" in value) {
        const envName = String(value.env || "");
        if (!envName || !(envName in process.env)) {
          fail(`Proxy rewrite rule ${index} requires env ${envName || "<empty>"}`);
        }
      } else if (!("value" in value)) {
        fail(`Proxy rewrite rule ${index} field ${field} requires env or value`);
      }
    }
    return { path, methods: normalizedMethods, set };
  });
}

const upstreamOrigin = (process.env.CONTEXTBENCH_PROXY_UPSTREAM_ORIGIN || "").trim();
if (!upstreamOrigin) {
  fail("CONTEXTBENCH_PROXY_UPSTREAM_ORIGIN is required");
}

const listenHost = (process.env.CONTEXTBENCH_PROXY_LISTEN_HOST || "127.0.0.1").trim();
const port = Number.parseInt(process.env.CONTEXTBENCH_PROXY_PORT || "", 10);
const healthPath = (process.env.CONTEXTBENCH_PROXY_HEALTH_PATH || "/health").trim();
const rewriteRules = parseRules();

if (!Number.isInteger(port) || port <= 0 || port > 65535) {
  fail(`Invalid CONTEXTBENCH_PROXY_PORT: ${process.env.CONTEXTBENCH_PROXY_PORT || ""}`);
}

function rewriteBody({ request, body }) {
  const requestUrl = new URL(request.url || "/", "http://127.0.0.1");
  const rule = rewriteRules.find(
    (candidate) => candidate.path === requestUrl.pathname && candidate.methods.includes(String(request.method || "").toUpperCase())
  );
  if (!rule) {
    return body;
  }

  const payload = JSON.parse(body.toString("utf8") || "{}");
  for (const [field, value] of Object.entries(rule.set)) {
    payload[field] = "env" in value ? process.env[String(value.env)] : value.value;
  }
  return Buffer.from(JSON.stringify(payload), "utf8");
}

const server = http.createServer((request, response) => {
  if (request.method === "GET" && request.url === healthPath) {
    response.writeHead(200, { "content-type": "text/plain" });
    response.end("ok\n");
    return;
  }

  const chunks = [];
  request.on("data", (chunk) => chunks.push(chunk));
  request.on("end", async () => {
    try {
      const incomingBody = Buffer.concat(chunks);
      let outgoingBody = incomingBody;
      const headers = { ...request.headers };
      delete headers.host;
      delete headers["content-length"];

      outgoingBody = rewriteBody({ request, body: incomingBody });
      if (outgoingBody !== incomingBody) {
        headers["content-type"] = "application/json";
      }

      const upstream = await fetch(new URL(request.url || "/", upstreamOrigin), {
        method: request.method,
        headers,
        body: request.method === "GET" || request.method === "HEAD" ? undefined : outgoingBody,
      });
      response.writeHead(upstream.status, Object.fromEntries(upstream.headers));
      response.end(Buffer.from(await upstream.arrayBuffer()));
    } catch (error) {
      response.writeHead(502, { "content-type": "text/plain" });
      response.end(String(error?.stack || error));
    }
  });
});

server.listen(port, listenHost, () => {
  console.error(`HTTP JSON rewrite proxy listening on ${listenHost}:${port}`);
});
