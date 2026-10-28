import { useEffect, useState } from "react";
import type { ExecutionBackend } from "../execution/ExecutionBackend.js";
import {
  RemoteExecutionBackend,
  type RemoteServerConfig,
} from "../execution/RemoteExecutionBackend.js";
import { WorkerExecutionBackend } from "../execution/WorkerExecutionBackend.js";
import { NumblWorkerClient } from "../numblWorkerClient.js";
import {
  useWorkerLifecycle,
  type UseNumblWorkerOptions,
} from "./useNumblWorker.js";

/**
 * The IDE's two execution backends: a worker owned by the calling component
 * (started on mount, terminated on unmount) and the execution server
 * configured by `remote`. `active` is the one runs should use.
 */
export function useExecutionBackends(
  remote: RemoteServerConfig & { enabled: boolean },
  workerOptions: UseNumblWorkerOptions
): { worker: WorkerExecutionBackend; active: ExecutionBackend } {
  const [worker] = useState(
    () => new WorkerExecutionBackend(new NumblWorkerClient({}, workerOptions))
  );
  useWorkerLifecycle(worker.client, workerOptions.optimization);

  const { serviceUrl, passkey } = remote;
  const [remoteBackend] = useState(
    () => new RemoteExecutionBackend({ serviceUrl, passkey })
  );
  useEffect(
    () => remoteBackend.configure({ serviceUrl, passkey }),
    [remoteBackend, serviceUrl, passkey]
  );

  return { worker, active: remote.enabled ? remoteBackend : worker };
}
