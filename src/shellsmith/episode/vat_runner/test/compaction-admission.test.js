import assert from "node:assert/strict";
import test from "node:test";
import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

import { AgentSession } from "@earendil-works/pi-coding-agent";

const require = createRequire(import.meta.resolve("@earendil-works/pi-coding-agent"));
const aiModule = require.resolve.paths("@earendil-works/pi-ai")
  .map((root) => join(root, "@earendil-works/pi-ai/dist/utils/overflow.js"))
  .find(existsSync);
const { isContextOverflow } = await import(pathToFileURL(aiModule).href);

test("exact rendered admission errors trigger one Pi compact-and-retry attempt", async () => {
  const failure = {
    role: "assistant", provider: "shellsmith-mlx", model: "shellsmith-player",
    stopReason: "error", timestamp: Date.now(),
    errorMessage: "422 context_length_exceeded: MLX prompt_tokens=32767, window=32768, available=1.",
    usage: { input: 0, cacheRead: 0, output: 0 },
  };
  assert.equal(isContextOverflow(failure, 32768), true);
  let attempts = 0;
  const emitted = [];
  const session = {
    settingsManager: { getCompactionSettings: () => ({ enabled: true, reserveTokens: 12288, keepRecentTokens: 8192 }) },
    model: { provider: failure.provider, id: failure.model, contextWindow: 32768, maxTokens: 8192 },
    sessionManager: { getBranch: () => [] },
    agent: { state: { messages: [failure] } },
    _overflowRecoveryAttempted: false,
    _emit: (event) => emitted.push(event),
    _runAutoCompaction: async (reason, willRetry) => {
      assert.equal(reason, "overflow");
      assert.equal(willRetry, true);
      attempts += 1;
      return true;
    },
  };
  assert.equal(await AgentSession.prototype._checkCompaction.call(session, failure), true);
  assert.equal(session.agent.state.messages.length, 0);
  assert.equal(await AgentSession.prototype._checkCompaction.call(session, failure), false);
  assert.equal(attempts, 1);
  assert.equal(emitted.at(-1).willRetry, false);
  assert.match(emitted.at(-1).errorMessage, /one compact-and-retry attempt/);
  assert.equal(isContextOverflow({ ...failure, errorMessage: "MLX native tool call malformed" }, 32768), false);
});
