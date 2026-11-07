/**
 * Real-workspace checks for the ephemeral vat.
 *
 * The simulated filesystem exists in the Game Master. Extra physical files
 * are inert because every Player tool crosses the wire. AGENTS.md is
 * the sole real context input, so it must remain unchanged.
 */

import { lstat, readFile, readdir } from "node:fs/promises";
import { isAbsolute, join } from "node:path";

export async function assertAgentsOnly(workspacePath, expectedContent = null) {
  if (!isAbsolute(workspacePath)) {
    throw new Error(`workspace_path must be absolute: ${workspacePath}`);
  }
  const entries = (await readdir(workspacePath)).sort();
  if (!entries.includes("AGENTS.md")) {
    throw new Error(`vat workspace must contain AGENTS.md; found ${JSON.stringify(entries)}`);
  }
  const agentsPath = join(workspacePath, "AGENTS.md");
  const stat = await lstat(agentsPath);
  if (!stat.isFile() || stat.isSymbolicLink()) {
    throw new Error("vat AGENTS.md must be a regular non-symlink file");
  }
  const content = await readFile(agentsPath, "utf8");
  if (content.trim().length === 0) {
    throw new Error("vat AGENTS.md must not be empty");
  }
  if (expectedContent !== null && content !== expectedContent) {
    throw new Error("vat AGENTS.md changed after Pi initialization");
  }
  return { agentsPath, content };
}

export async function assertEmptyAgentHome(agentHomePath) {
  if (!isAbsolute(agentHomePath)) {
    throw new Error(`agent_home_path must be absolute: ${agentHomePath}`);
  }
  const entries = await readdir(agentHomePath);
  if (entries.length !== 0) {
    throw new Error(
      `isolated Pi agent home must begin empty; found ${JSON.stringify(entries.sort())}`,
    );
  }
}
