/**
 * Suspends Pi tool executors until Python returns simulated observations.
 *
 * Architecture map (Tony sibling policy):
 *   message_end -> DECLARED (no world emit yet)
 *       -> Pi validates each sibling
 *       -> VALIDATED when executor registers, or HARNESS_REJECTED on schema fail
 *       -> DISPATCHED: emit tool_request messages for the whole sibling set
 *       -> hold all Player-facing results until every world slot returns
 *       -> CONSUMED after every sibling executor settles
 *
 * Why this exists:
 *   World mutation must not begin before Pi validation. Valid siblings still
 *   receive world observations. Harness-invalid siblings get an ordinary
 *   SimulatedToolError to Player; the world may still score a no-op turn.
 *   Player-facing results are held until the sibling batch artifact is complete.
 */

function normalizeContent(content) {
  if (typeof content !== "string") {
    throw new Error("tool_response.content must be a string");
  }
  return [{ type: "text", text: content }];
}

function sameArguments(left, right) {
  return JSON.stringify(left) === JSON.stringify(right);
}

export class SimulatedToolError extends Error {
  constructor(content) {
    const blocks = normalizeContent(content);
    const text = blocks
      .filter((block) => block.type === "text" && typeof block.text === "string")
      .map((block) => block.text)
      .join("\n");
    super(text);
    this.name = "SimulatedToolError";
  }
}

const HARNESS_REJECT_PREFIX = "Tool call rejected by harness validation";

export class ToolBridge {
  constructor(send, onTerminate = () => {}) {
    this.send = send;
    this.onTerminate = onTerminate;
    this.entries = new Map();
    this.batches = new Set();
    this.closed = false;
  }

  /**
   * Record sibling membership without talking to the world yet.
   *
   * Main caller:
   *   runner.js's synchronous message_end subscription.
   */
  declareBatch(toolCalls) {
    if (this.closed) throw new Error("vat tool bridge is closed");
    if (!Array.isArray(toolCalls) || toolCalls.length === 0) return;

    const ids = toolCalls.map((call) => call.id);
    if (new Set(ids).size !== ids.length) {
      throw new Error("assistant action contains duplicate tool_call_id values");
    }
    if (ids.some((id) => this.entries.has(id))) {
      throw new Error("assistant action reuses a pending tool_call_id");
    }

    const batch = {
      ids,
      worldResolved: new Set(),
      terminateRequested: false,
      finished: false,
      dispatched: false,
      released: false,
    };
    this.batches.add(batch);
    for (const call of toolCalls) {
      this.entries.set(call.id, {
        batch,
        toolName: call.name,
        arguments: call.arguments,
        state: "declared",
        consumer: null,
        worldOutcome: null,
        harnessRejectReason: null,
      });
    }
  }

  /**
   * Mark a declared sibling as harness-invalid before its executor runs.
   *
   * Main caller:
   *   runner.js on Pi tool_execution_end with isError before executor registration.
   */
  /**
   * Whether a tool call is still awaiting Pi validation (state "declared").
   *
   * Main caller:
   *   runner.js on tool_execution_end isError, to distinguish a genuine Pi
   *   schema/harness rejection (executor never ran) from the normal error
   *   round-trip of a world-simulated is_error result (SimulatedToolError),
   *   whose entry is already validated/dispatched or settled and removed.
   */
  isPendingDeclared(toolCallId) {
    const entry = this.entries.get(toolCallId);
    return Boolean(entry && entry.state === "declared");
  }

  markHarnessRejected(toolCallId, reason) {
    if (this.closed) throw new Error("vat tool bridge is closed");
    const entry = this.entries.get(toolCallId);
    if (!entry) {
      throw new Error(`harness reject for unknown tool_call_id: ${toolCallId}`);
    }
    if (entry.state === "validated" || entry.state === "dispatched") {
      throw new Error(
        `cannot harness-reject tool_call_id after validation: ${toolCallId}`,
      );
    }
    if (entry.state === "harness_rejected") return;
    entry.state = "harness_rejected";
    entry.harnessRejectReason =
      typeof reason === "string" && reason.trim()
        ? reason.trim()
        : `${HARNESS_REJECT_PREFIX}: ${toolCallId}`;
    this.#maybeDispatch(entry.batch);
  }

  request({ toolCallId, toolName, arguments: args, signal }) {
    if (this.closed) {
      return Promise.reject(new Error("vat tool bridge is closed"));
    }

    // Direct unit callers without message_end still form an explicit one-call batch.
    if (!this.entries.has(toolCallId)) {
      this.declareBatch([{ id: toolCallId, name: toolName, arguments: args }]);
    }
    const entry = this.entries.get(toolCallId);
    if (entry.consumer !== null) {
      return Promise.reject(new Error(`duplicate tool executor: ${toolCallId}`));
    }
    if (entry.state === "harness_rejected") {
      return Promise.reject(new SimulatedToolError(entry.harnessRejectReason));
    }
    if (entry.toolName !== toolName || !sameArguments(entry.arguments, args)) {
      return Promise.reject(
        new Error(`declared tool call does not match executor: ${toolCallId}`),
      );
    }

    entry.state = "validated";
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        reject(new Error(`tool call aborted: ${toolCallId}`));
      };
      if (signal?.aborted) {
        onAbort();
        return;
      }
      signal?.addEventListener("abort", onAbort, { once: true });
      entry.consumer = {
        resolve,
        reject,
        cleanup: () => signal?.removeEventListener("abort", onAbort),
      };
      this.#maybeDispatch(entry.batch);
      this.#maybeRelease(entry.batch);
    });
  }

  resolve(message) {
    const entry = this.entries.get(message.tool_call_id);
    if (!entry || entry.worldOutcome !== null) {
      throw new Error(
        `stale or unknown tool_response: ${message.tool_call_id}`,
      );
    }
    entry.worldOutcome = {
      content: normalizeContent(message.content ?? ""),
      details: message.details ?? {},
      isError: message.is_error ?? false,
      terminate: message.terminate ?? false,
      rawContent: message.content ?? "",
      omitPlayerFacing: Boolean(message.omit_player_facing),
    };
    if (entry.worldOutcome.terminate) {
      entry.batch.terminateRequested = true;
    }
    entry.batch.worldResolved.add(message.tool_call_id);
    this.#maybeRelease(entry.batch);
  }

  #maybeDispatch(batch) {
    if (batch.dispatched || batch.finished) return;
    const entries = batch.ids.map((id) => this.entries.get(id));
    if (entries.some((entry) => entry == null)) {
      throw new Error("sibling batch is missing a declared entry");
    }
    const ready = entries.every(
      (entry) =>
        entry.state === "validated" || entry.state === "harness_rejected",
    );
    if (!ready) return;

    batch.dispatched = true;
    for (const id of batch.ids) {
      const entry = this.entries.get(id);
      const harnessRejected = entry.state === "harness_rejected";
      if (!harnessRejected) {
        entry.state = "dispatched";
      }
      this.send({
        type: "tool_request",
        tool_call_id: id,
        tool_name: entry.toolName,
        arguments: entry.arguments,
        harness_rejected: harnessRejected,
        harness_reject_reason: harnessRejected
          ? entry.harnessRejectReason
          : null,
      });
    }
  }

  #maybeRelease(batch) {
    if (batch.released || batch.finished) return;
    if (!batch.dispatched) return;
    if (batch.worldResolved.size !== batch.ids.length) return;

    // Hold every Player-facing result until the whole sibling turn is complete.
    for (const id of batch.ids) {
      const entry = this.entries.get(id);
      if (entry.state === "harness_rejected") {
        // Pi never registered an executor for schema rejects; nothing to settle.
        // If an executor somehow registered, reject it with the harness error.
        if (entry.consumer !== null) {
          entry.consumer.cleanup();
          entry.consumer.reject(
            new SimulatedToolError(entry.harnessRejectReason),
          );
          entry.consumer = null;
        }
        continue;
      }
      if (entry.consumer === null) {
        throw new Error(
          `world responded before Pi executor registered: ${id}`,
        );
      }
      const outcome = entry.worldOutcome;
      entry.consumer.cleanup();
      if (outcome.omitPlayerFacing || outcome.isError) {
        entry.consumer.reject(new SimulatedToolError(outcome.rawContent));
      } else {
        entry.consumer.resolve({
          content: outcome.content,
          details: outcome.details,
          terminate: outcome.terminate,
        });
      }
      entry.consumer = null;
    }

    batch.released = true;
    batch.finished = true;
    for (const id of batch.ids) this.entries.delete(id);
    this.batches.delete(batch);
    if (batch.terminateRequested) {
      queueMicrotask(() => this.onTerminate());
    }
  }

  close(reason = "vat tool bridge closed") {
    this.closed = true;
    for (const [toolCallId, entry] of this.entries.entries()) {
      entry.consumer?.cleanup();
      entry.consumer?.reject(new Error(`${reason}: ${toolCallId}`));
    }
    this.entries.clear();
    this.batches.clear();
  }
}
