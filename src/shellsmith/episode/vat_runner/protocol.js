/**
 * Strict LF-delimited protocol helpers for the Node side of the vat boundary.
 *
 * Architecture:
 *   stdin chunks -> exact LF framing -> JSON parse -> host-message validation
 *   runner event -> JSON serialization -> one atomic stdout line
 *
 * Stdout is reserved exclusively for protocol records. Diagnostics belong on
 * stderr so Python never mistakes logging for a tool response.
 */

export const PROTOCOL_VERSION = 1;

function requireRecord(value, context) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new Error(`${context} must be a JSON object`);
  }
  return value;
}

function requireString(record, key, context) {
  const value = record[key];
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`${context}.${key} must be a non-empty string`);
  }
  return value;
}

function requireStringArray(record, key, context) {
  const value = record[key];
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string" || item.length === 0)) {
    throw new Error(`${context}.${key} must be a non-empty string array`);
  }
  return value;
}

export function validateHostMessage(value) {
  const record = requireRecord(value, "vat message");
  if (record.protocol_version !== PROTOCOL_VERSION) {
    throw new Error(
      `protocol version mismatch: expected ${PROTOCOL_VERSION}, got ${String(record.protocol_version)}`,
    );
  }
  const type = requireString(record, "type", "vat message");

  if (type === "initialize") {
    requireString(record, "request_id", type);
    requireString(record, "episode_id", type);
    requireString(record, "workspace_path", type);
    requireString(record, "agent_home_path", type);
    requireStringArray(record, "enabled_tools", type);
    const model = requireRecord(record.model, "initialize.model");
    requireString(model, "provider", "initialize.model");
    requireString(model, "id", "initialize.model");
    requireString(model, "base_url", "initialize.model");
    requireString(model, "api", "initialize.model");
    if (!Number.isInteger(model.context_window) || model.context_window <= 0) {
      throw new Error("initialize.model.context_window must be a positive integer");
    }
    if (!Number.isInteger(model.max_tokens) || model.max_tokens <= 0) {
      throw new Error("initialize.model.max_tokens must be a positive integer");
    }
    if (typeof model.reasoning !== "boolean") {
      throw new Error("initialize.model.reasoning must be a boolean");
    }
    if (model.compaction !== undefined) {
      const settings = requireRecord(model.compaction, "initialize.model.compaction");
      for (const key of ["reserve_tokens", "keep_recent_tokens"]) {
        if (!Number.isInteger(settings[key]) || settings[key] <= 0) {
          throw new Error(`initialize.model.compaction.${key} must be a positive integer`);
        }
      }
      if (settings.reserve_tokens >= model.context_window ||
          settings.keep_recent_tokens >= model.context_window - settings.reserve_tokens) {
        throw new Error("initialize.model.compaction budgets must fit the context window");
      }
    }
    requireString(model, "thinking_level", "initialize.model");
    if (model.api_key_env !== undefined) {
      requireString(model, "api_key_env", "initialize.model");
    }
    if (record.pi_sdk_module_path !== undefined) {
      requireString(record, "pi_sdk_module_path", "initialize");
    }
    if (record.retry !== undefined) {
      const retry = requireRecord(record.retry, "initialize.retry");
      if (typeof retry.player_enabled !== "boolean") throw new Error("initialize.retry.player_enabled must be boolean");
      for (const key of ["player_max_retries", "player_base_delay_ms", "provider_max_retry_delay_ms"]) {
        if (!Number.isInteger(retry[key]) || retry[key] < (key === "player_max_retries" ? 0 : 1)) {
          throw new Error(`initialize.retry.${key} must be a valid retry limit`);
        }
      }
    }
    return record;
  }

  if (type === "prompt") {
    requireString(record, "request_id", type);
    requireString(record, "text", type);
    return record;
  }

  if (type === "tool_response") {
    requireString(record, "tool_call_id", type);
    if (!("content" in record)) {
      throw new Error("tool_response.content is required");
    }
    if (typeof record.is_error !== "boolean") {
      throw new Error("tool_response.is_error must be a boolean");
    }
    if (record.terminate !== undefined && typeof record.terminate !== "boolean") {
      throw new Error("tool_response.terminate must be a boolean");
    }
    if (record.details !== undefined) {
      requireRecord(record.details, "tool_response.details");
    }
    return record;
  }

  if (type === "shutdown") {
    requireString(record, "request_id", type);
    return record;
  }

  throw new Error(`unknown host message type: ${type}`);
}

export async function* readJsonl(stream) {
  let pending = Buffer.alloc(0);
  for await (const chunk of stream) {
    pending = Buffer.concat([pending, Buffer.from(chunk)]);
    let newlineIndex = pending.indexOf(0x0a);
    while (newlineIndex >= 0) {
      const line = pending.subarray(0, newlineIndex);
      pending = pending.subarray(newlineIndex + 1);
      if (line.length === 0) {
        throw new Error("vat protocol does not allow blank lines");
      }
      let parsed;
      try {
        parsed = JSON.parse(line.toString("utf8"));
      } catch (error) {
        throw new Error(`malformed vat protocol JSON: ${error.message}`, { cause: error });
      }
      yield validateHostMessage(parsed);
      newlineIndex = pending.indexOf(0x0a);
    }
  }
  if (pending.length !== 0) {
    throw new Error("vat protocol ended with a non-LF-terminated record");
  }
}

export function writeJsonl(stream, message) {
  const record = {
    protocol_version: PROTOCOL_VERSION,
    ...message,
  };
  stream.write(`${JSON.stringify(record)}\n`);
}
