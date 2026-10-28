import { useEffect, useState } from "react";
import type { OptLevel } from "../numbl-core/executors/plugins.js";
import {
  NumblWorkerClient,
  type NumblWorkerHandlers,
  type NumblWorkerOptions,
} from "../numblWorkerClient.js";

export interface UseNumblWorkerOptions extends NumblWorkerOptions {
  /** Default optimization level, sent whenever it changes. */
  optimization?: OptLevel;
}

/**
 * A numbl-worker owned by the calling component: started on mount and
 * terminated on unmount. Handlers may change between renders; the latest
 * ones receive each message. Options other than `optimization` are read once.
 */
export function useNumblWorker(
  handlers: NumblWorkerHandlers,
  options: UseNumblWorkerOptions = {}
): NumblWorkerClient {
  const [client] = useState(() => new NumblWorkerClient({}, options));
  useEffect(() => client.setHandlers(handlers));

  useEffect(() => {
    client.start();
    return () => client.terminate();
  }, [client]);

  const { optimization } = options;
  useEffect(() => {
    if (optimization) client.setOptimization(optimization);
  }, [client, optimization]);

  return client;
}
