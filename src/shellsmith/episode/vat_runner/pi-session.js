/**
 * Constructs the one real Pi AgentSession used by an episode.
 *
 * Resource discovery is narrowed to the generated AGENTS.md. Settings and
 * conversation state stay in memory. Built-in executors are disabled and
 * replaced by the wire-backed definitions from virtual-tools.js.
 *
 * Architecture map
 * ----------------
 *   resolve Pi SDK (file URL to dist/index.js, or package name)
 *     -> SettingsManager.inMemory + DefaultResourceLoader (AGENTS.md only)
 *     -> ModelRuntime.create with in-memory CredentialStore (no auth.json)
 *     -> registerProvider + setRuntimeApiKey for Player/Tinker endpoint
 *     -> createAgentSession(modelRuntime, virtual tools, in-memory session)
 *
 * Why ModelRuntime (not AuthStorage / ModelRegistry):
 *   Pi 0.80.8 removed public AuthStorage / modelRegistry session options and
 *   stopped exporting AuthStorage.inMemory. Coding-agent auth now lives on
 *   ModelRuntime. AuthStorage.inMemory still exists as a private CredentialStore
 *   implementation next to dist/index.js; we deep-import it so the vat never
 *   writes ~/.pi/agent/auth.json.
 */

import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import { buildVirtualTools } from "./virtual-tools.js";
import { assertAgentsOnly, assertEmptyAgentHome } from "./workspace.js";

async function loadInMemoryCredentialStore(piModuleHref) {
  /**
   * Load AuthStorage.inMemory from the SDK's private core module.
   *
   * Main callers:
   *   createPiVatSession, before ModelRuntime.create.
   *
   * Why this exists:
   *   Public package exports no longer include AuthStorage. The class still
   *   ships at dist/core/auth-storage.js and implements pi-ai CredentialStore,
   *   which ModelRuntime.create({ credentials }) accepts.
   */
  if (!piModuleHref.startsWith("file:")) {
    throw new Error(
      "Pi SDK must be loaded via an absolute file URL (dist/index.js) so the " +
        "vat can deep-import in-memory AuthStorage. Pass --pi-sdk-module-path.",
    );
  }
  const indexPath = fileURLToPath(piModuleHref);
  const authHref = pathToFileURL(
    join(dirname(indexPath), "core", "auth-storage.js"),
  ).href;
  const { AuthStorage } = await import(authHref);
  if (typeof AuthStorage?.inMemory !== "function") {
    throw new Error(
      `Pi AuthStorage.inMemory is missing at ${authHref}. Upgrade or pin ` +
        "@earendil-works/pi-coding-agent >= 0.80.8.",
    );
  }
  return AuthStorage.inMemory();
}

async function configurePiHttpTimeout(piModuleHref, config) {
  /** Keep a serialized model request open for the episode's prompt budget.
   *
   * Main callers:
   *   createPiVatSession, before any model request can start.
   *
   * Why this exists:
   *   Pi's five-minute default can expire during queued inference or long
   *   generation. Both its dispatcher and SDK request options must receive
   *   the episode's configured deadline.
   */
  if (!piModuleHref.startsWith("file:")) {
    throw new Error(
      "Pi SDK must be loaded via an absolute file URL (dist/index.js) so the " +
        "vat can configure its HTTP dispatcher. Pass --pi-sdk-module-path.",
    );
  }
  const indexPath = fileURLToPath(piModuleHref);
  const dispatcherHref = pathToFileURL(
    join(dirname(indexPath), "core", "http-dispatcher.js"),
  ).href;
  const { configureHttpDispatcher } = await import(dispatcherHref);
  if (typeof configureHttpDispatcher !== "function") {
    throw new Error(
      `Pi configureHttpDispatcher is missing at ${dispatcherHref}. ` +
        "Upgrade or pin @earendil-works/pi-coding-agent >= 0.84.1.",
    );
  }
  // Match the dispatcher deadline to the episode's configured prompt budget.
  // A hardcoded 15-minute cap silently defeats the outer prompt_timeout_seconds
  // (43200s by default) and aborts long, otherwise valid Player
  // prompts. Fall back to the historical 900s (900_000 ms) when unset.
  const promptSeconds = Number(config && config.prompt_timeout_seconds) || 900;
  const timeoutMs = Math.round(promptSeconds * 1000);
  configureHttpDispatcher(timeoutMs);
  return timeoutMs;
}

export async function createPiVatSession(config, bridge) {
  const piModule = config.pi_sdk_module_path
    ? pathToFileURL(config.pi_sdk_module_path).href
    : "@earendil-works/pi-coding-agent";
  const [{ content: agents }, pi] = await Promise.all([
    assertAgentsOnly(config.workspace_path),
    import(piModule).catch((error) => {
      throw new Error(
        "Pi SDK is unavailable. Run npm install in vat_runner before a live episode.",
        { cause: error },
      );
    }),
  ]);
  await assertEmptyAgentHome(config.agent_home_path);
  const httpIdleTimeoutMs = await configurePiHttpTimeout(piModule, config);

  if (typeof pi.ModelRuntime?.create !== "function") {
    throw new Error(
      "Pi SDK is too old for ModelRuntime.create. Need @earendil-works/" +
        "pi-coding-agent >= 0.80.8 (current global broke AuthStorage.inMemory).",
    );
  }

  const settingsManager = pi.SettingsManager.inMemory({
    // createAgentSession separately builds provider timeoutMs from this
    // setting. Configuring only the dispatcher leaves a five-minute SDK cap.
    httpIdleTimeoutMs,
    retry: {
      enabled: config.retry?.player_enabled ?? true,
      maxRetries: config.retry?.player_max_retries ?? 3,
      baseDelayMs: config.retry?.player_base_delay_ms ?? 2000,
      provider: {maxRetryDelayMs: config.retry?.provider_max_retry_delay_ms ?? 60000},
    },
    // Pi estimates tokens before the endpoint renders the native Qwen prompt.
    // Each resolved Player profile reserves room for estimator error and its
    // completion budget; legacy callers retain their established 64k/20k.
    compaction: {
      enabled: true,
      reserveTokens: config.model.compaction?.reserve_tokens ?? 65536,
      keepRecentTokens: config.model.compaction?.keep_recent_tokens ?? 20000,
    },
  });
  const resourceLoader = new pi.DefaultResourceLoader({
    cwd: config.workspace_path,
    agentDir: config.agent_home_path,
    settingsManager,
    // Pi normally discovers resources from the project tree and the user's
    // agent directory. The vat must instead see exactly one generated context
    // file and no host extensions, skills, templates, themes, or ancestor
    // AGENTS.md files.
    noExtensions: true,
    noSkills: true,
    noPromptTemplates: true,
    noThemes: true,
    noContextFiles: true,
    agentsFilesOverride: () => ({
      agentsFiles: [
        {
          path: join(config.workspace_path, "AGENTS.md"),
          content: agents,
        },
      ],
    }),
  });
  await resourceLoader.reload();

  const apiKey = config.model.api_key_env
    ? process.env[config.model.api_key_env]
    : undefined;
  if (config.model.api_key_env && !apiKey) {
    throw new Error(
      `Required model credential environment variable is absent: ${config.model.api_key_env}`,
    );
  }

  const credentials = await loadInMemoryCredentialStore(piModule);
  const modelRuntime = await pi.ModelRuntime.create({
    credentials,
    modelsPath: null,
    // Pi 0.80.x controls later credential-refresh networking with this
    // construction option. Newer Pi versions additionally accept the explicit
    // per-refresh option passed to setRuntimeApiKey below.
    allowModelNetwork: false,
  });
  if (apiKey) {
    // The vat's only model endpoint is the explicitly registered local Tinker
    // proxy below. Pi otherwise refreshes every built-in provider catalog when
    // a runtime key is set; that unrelated network request can hang startup and
    // prevents the runner from emitting its `initialized` response.
    await modelRuntime.setRuntimeApiKey(config.model.provider, apiKey, {
      allowNetwork: false,
    });
  }
  modelRuntime.registerProvider(config.model.provider, {
    baseUrl: config.model.base_url,
    api: config.model.api,
    // Dynamic providers require an auth declaration while loading. This value
    // exists only inside the short-lived Node process; protocol and audit logs
    // contain only the environment-variable name.
    apiKey,
    authHeader: true,
    models: [
      {
        id: config.model.id,
        name: config.model.id,
        reasoning: config.model.reasoning,
        input: ["text"],
        contextWindow: config.model.context_window,
        maxTokens: config.model.max_tokens,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: {
          supportsDeveloperRole: false,
          // Shellsmith RFT requires real thinking on the wire. When this
          // is false, Pi never sends reasoning_effort and the proxy falls back
          // to the non-thinking renderer.
          supportsReasoningEffort: true,
        },
      },
    ],
  });
  const model = modelRuntime.getModel(config.model.provider, config.model.id);
  if (!model) {
    throw new Error(
      `Pi model registration failed: ${config.model.provider}/${config.model.id}.`,
    );
  }

  const virtualTools = buildVirtualTools({
    pi,
    cwd: config.workspace_path,
    bridge,
    enabledTools: config.enabled_tools,
  });
  const { session, extensionsResult } = await pi.createAgentSession({
    cwd: config.workspace_path,
    agentDir: config.agent_home_path,
    model,
    modelRuntime,
    thinkingLevel: config.model.thinking_level,
    noTools: "builtin",
    tools: config.enabled_tools,
    customTools: virtualTools,
    resourceLoader,
    sessionManager: pi.SessionManager.inMemory(config.workspace_path),
    settingsManager,
  });
  if (extensionsResult.errors.length > 0) {
    session.dispose();
    throw new Error(
      `Pi resource initialization failed: ${JSON.stringify(extensionsResult.errors)}`,
    );
  }
  return session;
}
