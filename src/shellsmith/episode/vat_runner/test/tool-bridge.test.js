import assert from "node:assert/strict";
import test from "node:test";

import { SimulatedToolError, ToolBridge } from "../tool-bridge.js";

test("declareBatch does not emit world requests before validation", () => {
  const emitted = [];
  const bridge = new ToolBridge((message) => emitted.push(message));
  bridge.declareBatch([
    { id: "call-1", name: "bash", arguments: { command: "pwd" } },
  ]);
  assert.deepEqual(emitted, []);
});

test("tool executor remains pending until its correlated response arrives", async () => {
  const emitted = [];
  const bridge = new ToolBridge((message) => emitted.push(message));

  const resultPromise = bridge.request({
    toolCallId: "call-1",
    toolName: "bash",
    arguments: { command: "pwd" },
  });
  assert.deepEqual(emitted, [
    {
      type: "tool_request",
      tool_call_id: "call-1",
      tool_name: "bash",
      arguments: { command: "pwd" },
      harness_rejected: false,
      harness_reject_reason: null,
    },
  ]);

  bridge.resolve({
    tool_call_id: "call-1",
    content: "/workspace\n",
    is_error: false,
  });

  assert.deepEqual(await resultPromise, {
    content: [{ type: "text", text: "/workspace\n" }],
    details: {},
    terminate: false,
  });
  assert.throws(
    () =>
      bridge.resolve({
        tool_call_id: "call-1",
        content: "stale",
        is_error: false,
      }),
    /stale or unknown/,
  );
});

test("simulated errors reject the Pi executor without local execution", async () => {
  const bridge = new ToolBridge(() => {});
  const resultPromise = bridge.request({
    toolCallId: "call-error",
    toolName: "read",
    arguments: { path: "missing.txt" },
  });

  bridge.resolve({
    tool_call_id: "call-error",
    content: "No such file or directory",
    is_error: true,
  });

  await assert.rejects(resultPromise, SimulatedToolError);
});

test("empty errors never manufacture simulator-specific prose", async () => {
  const bridge = new ToolBridge(() => {});
  const resultPromise = bridge.request({
    toolCallId: "call-empty-error",
    toolName: "bash",
    arguments: { command: "false" },
  });

  bridge.resolve({
    tool_call_id: "call-empty-error",
    content: "",
    is_error: true,
  });

  await assert.rejects(resultPromise, (error) => {
    assert.equal(error.message, "");
    assert.doesNotMatch(String(error), /simulated tool call failed/i);
    return true;
  });
});

test("terminal world response resolves the tool then requests Pi termination", async () => {
  let terminated = false;
  const bridge = new ToolBridge(
    () => {},
    () => {
      terminated = true;
    },
  );
  const resultPromise = bridge.request({
    toolCallId: "call-terminal",
    toolName: "bash",
    arguments: { command: "dangerous-command" },
  });

  bridge.resolve({
    tool_call_id: "call-terminal",
    content: "Operation prohibited",
    is_error: false,
    terminate: true,
  });

  const result = await resultPromise;
  assert.equal(result.terminate, true);
  await new Promise((resolve) => queueMicrotask(resolve));
  assert.equal(terminated, true);
});

test("validated siblings emit together; Player results wait for full world batch", async () => {
  const emitted = [];
  let terminated = false;
  const bridge = new ToolBridge(
    (message) => emitted.push(message),
    () => {
      terminated = true;
    },
  );
  bridge.declareBatch([
    { id: "call-a", name: "read", arguments: { path: "a" } },
    { id: "call-b", name: "read", arguments: { path: "b" } },
  ]);
  assert.deepEqual(emitted, []);

  const first = bridge.request({
    toolCallId: "call-a",
    toolName: "read",
    arguments: { path: "a" },
  });
  assert.deepEqual(emitted, []);

  const second = bridge.request({
    toolCallId: "call-b",
    toolName: "read",
    arguments: { path: "b" },
  });
  assert.deepEqual(
    emitted.map((message) => message.tool_call_id),
    ["call-a", "call-b"],
  );

  bridge.resolve({
    tool_call_id: "call-a",
    content: "A",
    is_error: false,
    terminate: false,
  });
  // First sibling must still be pending until the whole world batch returns.
  let firstSettled = false;
  first.then(() => {
    firstSettled = true;
  });
  await Promise.resolve();
  assert.equal(firstSettled, false);

  bridge.resolve({
    tool_call_id: "call-b",
    content: "B",
    is_error: false,
    terminate: true,
  });

  assert.equal((await first).content[0].text, "A");
  assert.equal((await second).content[0].text, "B");
  await new Promise((resolve) => queueMicrotask(resolve));
  assert.equal(terminated, true);
});

test("one valid and one harness-invalid sibling still dispatch; invalid gets harness error", async () => {
  const emitted = [];
  const bridge = new ToolBridge((message) => emitted.push(message));
  bridge.declareBatch([
    { id: "call-ok", name: "read", arguments: { path: "ok.txt" } },
    { id: "call-bad", name: "read", arguments: { file_path: "bad.txt" } },
  ]);

  const okPromise = bridge.request({
    toolCallId: "call-ok",
    toolName: "read",
    arguments: { path: "ok.txt" },
  });
  bridge.markHarnessRejected(
    "call-bad",
    "Tool call rejected by harness validation: missing required path",
  );

  assert.equal(emitted.length, 2);
  assert.equal(emitted[0].harness_rejected, false);
  assert.equal(emitted[1].harness_rejected, true);

  bridge.resolve({
    tool_call_id: "call-ok",
    content: "ok",
    is_error: false,
  });
  bridge.resolve({
    tool_call_id: "call-bad",
    content: "",
    is_error: true,
    omit_player_facing: true,
  });

  assert.equal((await okPromise).content[0].text, "ok");
});
