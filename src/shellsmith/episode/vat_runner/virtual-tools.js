/**
 * Builds Pi-compatible tool definitions whose executors cross the vat wire.
 *
 * We reuse Pi's names, descriptions, parameter schemas, and render metadata, but
 * replace every execute function. The base executor is never called. Sequential
 * execution gives the persistent Game Master one unambiguous state order.
 */

const FACTORY_NAMES = {
  read: "createReadTool",
  bash: "createBashTool",
  edit: "createEditTool",
  write: "createWriteTool",
  grep: "createGrepTool",
  find: "createFindTool",
  ls: "createLsTool",
};

// Pi only includes custom tools in the textual `Available tools` section when
// they opt in with `promptSnippet`. Our wire-backed tools deliberately replace
// Pi's built-ins, so inheriting the factory metadata alone is insufficient: the
// model receives valid API schemas but is told in prose that no tools exist.
// That mismatch is especially damaging for text-rendered Tinker/Qwen models,
// which then invent familiar aliases such as `read_file` or `list_directory`.
const PROMPT_SNIPPETS = {
  read: "Read a file by path, optionally selecting a line range",
  bash: "Run a shell command in the current workspace",
  edit: "Replace exact text in an existing file",
  write: "Create or overwrite a file with supplied content",
  grep: "Search file contents for a pattern",
  find: "Find files matching a glob pattern",
  ls: "List the entries in a directory",
};

export function buildVirtualTools({ pi, cwd, bridge, enabledTools }) {
  return enabledTools.map((toolName) => {
    const factoryName = FACTORY_NAMES[toolName];
    if (!factoryName) {
      throw new Error(
        `No Pi-compatible virtual tool factory is registered for ${toolName}`,
      );
    }
    const factory = pi[factoryName];
    if (typeof factory !== "function") {
      throw new Error(`Installed Pi SDK does not export ${factoryName}`);
    }
    const baseTool = factory(cwd);
    return {
      ...baseTool,
      promptSnippet: PROMPT_SNIPPETS[toolName],
      promptGuidelines: [
        `Use only the exact tool name \`${toolName}\`; do not substitute aliases.`,
      ],
      executionMode: "parallel",
      async execute(toolCallId, params, signal) {
        return bridge.request({
          toolCallId,
          toolName,
          arguments: params,
          signal,
        });
      },
    };
  });
}
