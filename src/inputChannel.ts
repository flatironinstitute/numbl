/**
 * input() over the blocking channel (numbl-core/runtime/blockingChannel.ts).
 * The worker sends an "input" request with the prompt and blocks; the host's
 * InputServer hands the prompt to its callback and answers with `respond`.
 * Stop wakes the worker, which drops the request.
 */

import {
  blockingChannelFor,
  createChannelServer,
  type ChannelReply,
  type ChannelServer,
} from "./numbl-core/runtime/blockingChannel.js";

/** Worker side: the `onInput` option for a run whose host answers input
 *  requests, or undefined without a cancel buffer. */
export function channelInput(
  cancelSAB: SharedArrayBuffer | undefined
): ((prompt: string) => string) | undefined {
  if (!cancelSAB) return undefined;
  const channel = blockingChannelFor(cancelSAB);
  return prompt => String(channel.request(self, "input", prompt) ?? "");
}

/** Host side: receives the worker's input requests. */
export class InputServer {
  private pending: ChannelReply | null = null;
  private readonly server: ChannelServer;

  constructor(onRequest: (prompt: string) => void) {
    this.server = createChannelServer({
      request: (cmd, body, reply) => {
        if (cmd !== "input") {
          reply.error(`unknown request '${cmd}'`);
          return;
        }
        this.pending = reply;
        onRequest(String(body ?? ""));
      },
      cancel: () => {
        this.pending = null;
      },
    });
  }

  /** Handle a worker message if it belongs to the channel. */
  handle(msg: unknown): boolean {
    return this.server.handle(msg);
  }

  /** Answer the pending input request (no-op if there is none). */
  respond(text: string): void {
    const reply = this.pending;
    this.pending = null;
    reply?.ok(text);
  }
}
