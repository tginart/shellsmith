import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, writeFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { ModelRuntime, SettingsManager } from "@earendil-works/pi-coding-agent";
import { createPiVatSession } from "../pi-session.js";

test("the real Pi SDK uses the episode deadline for provider requests", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "shellsmith-http-deadline-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const workspace = join(root, "workspace");
  const agentHome = join(root, "agent-home");
  await mkdir(workspace);
  await mkdir(agentHome);
  await writeFile(join(workspace, "AGENTS.md"), "# Isolated deadline probe\n");
  const requests = [];
  const managers = [];
  const inMemory = SettingsManager.inMemory;
  t.mock.method(SettingsManager, "inMemory", (settings) => {
    const manager = inMemory.call(SettingsManager, settings);
    managers.push(manager);
    return manager;
  });
  t.mock.method(ModelRuntime.prototype, "streamSimple", (_model, _context, options) => {
    requests.push(options);
    return "request-intercepted";
  });
  for (const [promptSeconds, expectedMs] of [[43200, 43200000], [undefined, 900000]]) {
    const session = await createPiVatSession({
      pi_sdk_module_path: fileURLToPath(import.meta.resolve("@earendil-works/pi-coding-agent")),
      workspace_path: workspace,
      agent_home_path: agentHome,
      prompt_timeout_seconds: promptSeconds,
      retry: {player_enabled: false, player_max_retries: 1, player_base_delay_ms: 1500,
              provider_max_retry_delay_ms: 30000},
      enabled_tools: ["read"],
      model: {
        provider: "shellsmith-deadline-test", id: "local-deadline-probe",
        base_url: "http://127.0.0.1:9/v1", api: "openai-completions",
        reasoning: false, context_window: 32768, max_tokens: 8192,
      },
    }, {});
    try {
      assert.equal(await session.agent.streamFunction(session.model, { messages: [] }, {}), "request-intercepted");
      assert.equal(requests.at(-1).timeoutMs, expectedMs);
      assert.deepEqual(managers.at(-1).getRetrySettings(), {enabled: false, maxRetries: 1, baseDelayMs: 1500});
      assert.equal(managers.at(-1).getProviderRetrySettings().maxRetryDelayMs, 30000);
    } finally {
      session.dispose();
    }
  }
});
