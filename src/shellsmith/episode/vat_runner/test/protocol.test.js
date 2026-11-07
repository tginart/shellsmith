import assert from "node:assert/strict";
import test from "node:test";
import { Readable } from "node:stream";

import { PROTOCOL_VERSION, readJsonl, validateHostMessage } from "../protocol.js";

test("strict JSONL decoder preserves records split across chunks", async () => {
  const first = JSON.stringify({
    protocol_version: PROTOCOL_VERSION,
    type: "prompt",
    request_id: "prompt-1",
    text: "Begin",
  });
  const second = JSON.stringify({
    protocol_version: PROTOCOL_VERSION,
    type: "shutdown",
    request_id: "shutdown-1",
  });
  const stream = Readable.from([first.slice(0, 13), `${first.slice(13)}\n${second}\n`]);

  const records = [];
  for await (const record of readJsonl(stream)) records.push(record);

  assert.equal(records.length, 2);
  assert.equal(records[0].type, "prompt");
  assert.equal(records[1].type, "shutdown");
});

test("decoder rejects unterminated and version-mismatched records", async () => {
  await assert.rejects(
    async () => {
      for await (const _ of readJsonl(Readable.from(['{"incomplete":true}']))) {
        // The generator fails at EOF.
      }
    },
    /non-LF-terminated/,
  );

  assert.throws(
    () =>
      validateHostMessage({
        protocol_version: 999,
        type: "shutdown",
        request_id: "x",
      }),
    /version mismatch/,
  );
});

test("small-context profiles carry bounded compaction settings", () => {
  const initialize = {
    protocol_version: PROTOCOL_VERSION, type: "initialize", request_id: "init", episode_id: "episode",
    workspace_path: "/workspace", agent_home_path: "/agent", enabled_tools: ["read"],
    model: { provider: "shellsmith-mlx", id: "player", base_url: "http://127.0.0.1",
      api: "openai-completions", context_window: 32768, max_tokens: 8192, reasoning: true,
      thinking_level: "high", compaction: { reserve_tokens: 12288, keep_recent_tokens: 8192 } },
  };
  assert.equal(validateHostMessage(initialize).model.compaction.reserve_tokens, 12288);
  for (const compaction of [
    { reserve_tokens: 65536, keep_recent_tokens: 20000 },
    { reserve_tokens: 12288, keep_recent_tokens: 22000 },
    { reserve_tokens: 12288, keep_recent_tokens: -1 },
  ]) {
    assert.throws(() => validateHostMessage({ ...initialize, model: { ...initialize.model, compaction } }), /compaction/);
  }
});
