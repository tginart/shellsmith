#!/usr/bin/env node

/**
 * Headless Pi-SDK vat runner.
 *
 * Architecture map:
 *   Python JSONL stdin
 *       -> initialize one isolated, in-memory Pi session
 *       -> accept one prompt at a time
 *       -> keep reading stdin while virtual tool Promises wait for responses
 *       -> emit only protocol JSONL on stdout
 *
 * A prompt is intentionally launched without awaiting it in the command loop.
 * This is what allows `tool_response` messages to resolve an executor while the
 * Pi agent loop remains suspended inside session.prompt().
 */

import { createPiVatSession } from "./pi-session.js";
import { readJsonl, writeJsonl } from "./protocol.js";
import { ToolBridge } from "./tool-bridge.js";
import { assertAgentsOnly } from "./workspace.js";

const send = (message) => writeJsonl(process.stdout, message);

let session = null;
let unsubscribe = null;
let activePrompt = null;
let workspacePath = null;
let agentsContent = null;
let terminationRequested = false;
const toolBridge = new ToolBridge(send, () => {
  terminationRequested = true;
  if (session !== null) {
    void session.abort().catch((error) => {
      send({
        type: "error",
        request_id: null,
        fatal: true,
        message: `failed to terminate Pi prompt: ${error instanceof Error ? error.message : String(error)}`,
      });
    });
  }
});

function eventProjection(event) {
  const payload = {};
  for (const key of ["turnIndex", "toolCallId", "toolName", "isError"]) {
    if (event[key] !== undefined) payload[key] = event[key];
  }
  if (event.type === "message_end" && event.message) {
    const content = Array.isArray(event.message.content)
      ? event.message.content.map((block) => {
          if (block.type === "text") return { type: "text", text: block.text };
          if (block.type === "thinking") {
            return { type: "thinking", thinking: block.thinking };
          }
          if (block.type === "toolCall") {
            return {
              type: "toolCall",
              id: block.id,
              name: block.name,
              arguments: block.arguments,
            };
          }
          return { type: String(block.type ?? "unknown") };
        })
      : event.message.content;
    payload.message = { role: event.message.role, content };
  }
  if (event.type === "turn_end" && event.message) {
    payload.assistant_role = event.message.role;
    payload.tool_result_count = Array.isArray(event.toolResults)
      ? event.toolResults.length
      : 0;
  }
  if (event.type === "agent_end") {
    payload.new_message_count = Array.isArray(event.messages)
      ? event.messages.length
      : 0;
    if (typeof event.willRetry === "boolean") {
      payload.will_retry = event.willRetry;
    }
  }
  if (event.type === "compaction_start") {
    payload.reason = event.reason;
  }
  if (event.type === "compaction_end") {
    payload.reason = event.reason;
    payload.aborted = event.aborted;
    payload.will_retry = event.willRetry;
    if (event.result) {
      payload.compaction = {
        summary: event.result.summary,
        first_kept_entry_id: event.result.firstKeptEntryId,
        tokens_before: event.result.tokensBefore,
        details: event.result.details ?? null,
      };
    }
  }
  return payload;
}

function latestAssistantText(messages) {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role !== "assistant" || !Array.isArray(message.content)) continue;
    return message.content
      .filter((block) => block.type === "text" && typeof block.text === "string")
      .map((block) => block.text)
      .join("\n");
  }
  return "";
}

async function initialize(message) {
  if (session !== null) throw new Error("vat runner is already initialized");
  agentsContent = (await assertAgentsOnly(message.workspace_path)).content;
  session = await createPiVatSession(message, toolBridge);
  workspacePath = message.workspace_path;
  unsubscribe = session.subscribe((event) => {
    // Streaming deltas can create thousands of wire records and do not affect
    // world state. Completed messages retain the same text/reasoning/tool call
    // content in one auditable event.
    if (event.type === "message_update") return;
    const payload = eventProjection(event);
    send({
      type: "agent_event",
      event_type: event.type,
      payload,
    });
    if (
      event.type === "message_end" &&
      payload.message?.role === "assistant" &&
      Array.isArray(payload.message.content)
    ) {
      // Declare sibling membership only. Do NOT emit world tool_request yet;
      // Pi must validate arguments first (see tool-bridge markHarnessRejected).
      const toolCalls = payload.message.content
        .filter((block) => block.type === "toolCall")
        .map((block) => ({
          id: block.id,
          name: block.name,
          arguments: block.arguments,
        }));
      toolBridge.declareBatch(toolCalls);
    }
    if (event.type === "tool_execution_end" && event.isError) {
      // Only a call still awaiting validation ("declared") is a genuine Pi
      // schema/harness rejection. A validated, dispatched, or already-settled
      // call reaching here is the ordinary error round-trip of a
      // world-simulated is_error observation (SimulatedToolError) — a normal
      // in-world outcome that must NOT be reclassified as a harness reject
      // (doing so used to throw and kill the episode).
      const toolCallId = event.toolCallId;
      if (
        typeof toolCallId === "string" &&
        toolCallId &&
        toolBridge.isPendingDeclared(toolCallId)
      ) {
        const reason =
          typeof event.error?.message === "string"
            ? event.error.message
            : typeof event.message === "string"
              ? event.message
              : `Tool call rejected by harness validation: ${toolCallId}`;
        toolBridge.markHarnessRejected(toolCallId, reason);
      }
    }
  });
  send({
    type: "initialized",
    request_id: message.request_id,
    session_id: session.sessionId,
    enabled_tools: message.enabled_tools,
  });
}

async function runPrompt(message) {
  terminationRequested = false;
  try {
    await assertAgentsOnly(workspacePath, agentsContent);
    await session.prompt(message.text);
    await assertAgentsOnly(workspacePath, agentsContent);
    send({
      type: "prompt_complete",
      request_id: message.request_id,
      final_text: latestAssistantText(session.messages),
      message_count: session.messages.length,
      terminated: terminationRequested,
    });
  } catch (error) {
    if (terminationRequested) {
      await assertAgentsOnly(workspacePath, agentsContent);
      send({
        type: "prompt_complete",
        request_id: message.request_id,
        final_text: latestAssistantText(session.messages),
        message_count: session.messages.length,
        terminated: true,
      });
    } else {
      send({
        type: "error",
        request_id: message.request_id,
        fatal: true,
        message: error instanceof Error ? error.message : String(error),
      });
    }
  } finally {
    activePrompt = null;
  }
}

async function dispatch(message) {
  if (message.type === "initialize") {
    await initialize(message);
    return true;
  }
  if (session === null) throw new Error("vat runner must be initialized first");

  if (message.type === "tool_response") {
    toolBridge.resolve(message);
    return true;
  }
  if (message.type === "prompt") {
    if (activePrompt !== null) throw new Error("overlapping vat prompts are forbidden");
    activePrompt = runPrompt(message);
    return true;
  }
  if (message.type === "shutdown") {
    if (activePrompt !== null) throw new Error("cannot shut down during an active prompt");
    unsubscribe?.();
    session.dispose();
    session = null;
    toolBridge.close("vat runner shutdown");
    send({ type: "shutdown_complete", request_id: message.request_id });
    return false;
  }
  throw new Error(`unsupported command: ${message.type}`);
}

try {
  for await (const message of readJsonl(process.stdin)) {
    try {
      const shouldContinue = await dispatch(message);
      if (!shouldContinue) break;
    } catch (error) {
      send({
        type: "error",
        request_id: typeof message.request_id === "string" ? message.request_id : null,
        fatal: true,
        message: error instanceof Error ? error.message : String(error),
      });
      break;
    }
  }
} catch (error) {
  send({
    type: "error",
    request_id: null,
    fatal: true,
    message: error instanceof Error ? error.message : String(error),
  });
} finally {
  toolBridge.close("vat runner input closed");
  unsubscribe?.();
  session?.dispose();
}
