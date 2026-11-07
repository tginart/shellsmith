import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { assertAgentsOnly, assertEmptyAgentHome } from "../workspace.js";

test("real vat allows inert files but rejects AGENTS.md mutation", async () => {
  const root = await mkdtemp(join(tmpdir(), "shellsmith-vat-node-test-"));
  const workspace = join(root, "workspace");
  const agentHome = join(root, "agent-home");
  await mkdir(workspace);
  await mkdir(agentHome);
  try {
    await writeFile(join(workspace, "AGENTS.md"), "# Public instructions\n", "utf8");
    const result = await assertAgentsOnly(workspace);
    assert.match(result.content, /Public instructions/);
    await assertEmptyAgentHome(agentHome);

    await writeFile(join(workspace, "invented.py"), "print('leak')\n", "utf8");
    await assertAgentsOnly(workspace, result.content);
    await writeFile(join(workspace, "AGENTS.md"), "# Changed\n", "utf8");
    await assert.rejects(
      assertAgentsOnly(workspace, result.content),
      /changed after Pi initialization/,
    );
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
