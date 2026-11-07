import assert from "node:assert/strict";
import test from "node:test";

import { buildVirtualTools } from "../virtual-tools.js";

test("virtual tools preserve Pi metadata but never invoke the base executor", async () => {
  let baseExecutions = 0;
  const pi = {
    createReadTool: () => ({
      name: "read",
      label: "Read",
      description: "Read a file",
      parameters: { type: "object" },
      execute: async () => {
        baseExecutions += 1;
      },
    }),
  };
  const bridged = [];
  const bridge = {
    request: async (request) => {
      bridged.push(request);
      return { content: [{ type: "text", text: "virtual" }], details: {} };
    },
  };

  const [tool] = buildVirtualTools({
    pi,
    cwd: "/tmp/vat/workspace",
    bridge,
    enabledTools: ["read"],
  });
  const result = await tool.execute("call-1", { path: "README.md" }, undefined);

  assert.equal(tool.description, "Read a file");
  assert.equal(
    tool.promptSnippet,
    "Read a file by path, optionally selecting a line range",
  );
  assert.deepEqual(tool.promptGuidelines, [
    "Use only the exact tool name `read`; do not substitute aliases.",
  ]);
  assert.equal(tool.executionMode, "parallel");
  assert.equal(baseExecutions, 0);
  assert.equal(bridged[0].toolName, "read");
  assert.deepEqual(result.content, [{ type: "text", text: "virtual" }]);
});

test("every wire-backed Pi tool opts into the textual system prompt", () => {
  const factoryNames = {
    read: "createReadTool",
    bash: "createBashTool",
    edit: "createEditTool",
    write: "createWriteTool",
    grep: "createGrepTool",
    find: "createFindTool",
    ls: "createLsTool",
  };
  const pi = Object.fromEntries(
    Object.entries(factoryNames).map(([toolName, factoryName]) => [
      factoryName,
      () => ({
        name: toolName,
        description: `${toolName} description`,
        parameters: { type: "object" },
      }),
    ]),
  );
  const tools = buildVirtualTools({
    pi,
    cwd: "/tmp/vat/workspace",
    bridge: { request: async () => ({ content: [], details: {} }) },
    enabledTools: Object.keys(factoryNames),
  });

  assert.deepEqual(
    tools.map((tool) => tool.name),
    ["read", "bash", "edit", "write", "grep", "find", "ls"],
  );
  for (const tool of tools) {
    assert.ok(tool.promptSnippet.length > 0);
    assert.match(tool.promptGuidelines[0], new RegExp(`\\b${tool.name}\\b`));
  }
});
