/**
 * Host side of the managed browser session. The worker code is inlined at
 * build time (see scripts/build_browser.ts) and started from a Blob URL, so
 * any bundler — or none — can consume this without worker-loading support.
 */
import workerCode from "./generated/worker-code.txt";
import type {
  BootFile,
  FromWorker,
  ToWorker,
  UihtmlComponent,
} from "./protocol.js";

export interface NumblSessionOptions {
  /** Project files. Relative paths land under /project/. */
  files: BootFile[];
  /** The script to run, as one of the paths in `files`. */
  mainFile: string;
  /**
   * Bootstrap the mip package manager (fetched from its GitHub release)
   * so the script can `mip load --install <pkg>`. Default true.
   */
  mip?: boolean;
  /**
   * Persist the /system directory (mip core + installed packages) in
   * IndexedDB so later page loads skip the downloads. Default true.
   */
  persistSystem?: boolean;
  /**
   * Wipe the persisted /system directory after this much inactivity, so a
   * stale mip core / package set refreshes eventually. Default 24 h.
   */
  systemInactivityMs?: number;
  optimization?: "0" | "1";
  maxIterations?: number;
  displayResults?: boolean;
  /** Console output from the script (fprintf/disp/mip logs). */
  onOutput?: (text: string) => void;
  /** Boot progress (package downloads, engine start). */
  onProgress?: (message: string) => void;
  /** A uihtml component was created/updated (its Data JSON-encoded). */
  onUihtml?: (compId: string, dataJson: string) => void;
  /** Script -> host events (MATLAB `sendEventToHTMLSource`). */
  onHtmlSourceEvent?: (compId: string, name: string, dataJson: string) => void;
}

export interface NumblSession {
  /** uihtml components the run created, in creation order. */
  readonly uihtmlComponents: readonly UihtmlComponent[];
  /** True if the run left a live uihtml session (events can be dispatched). */
  readonly hasUihtmlSession: boolean;
  /** Write a file into the session VFS (visible to later dispatched events). */
  writeFile(path: string, content: string | Uint8Array): void;
  /**
   * Read a file from the session VFS (e.g. results the script wrote).
   * Relative paths resolve under /project/. Rejects if the file is missing.
   */
  readFile(path: string): Promise<Uint8Array>;
  /**
   * Fire the script's HTMLEventReceivedFcn (host -> MATLAB). Resolves when
   * the callback returns; rejects if it errors (the session stays usable).
   */
  dispatchHtmlEvent(compId: string, name: string, data: unknown): Promise<void>;
  /** Terminate the worker. The session is unusable afterwards. */
  dispose(): void;
}

class NumblSessionImpl implements NumblSession {
  uihtmlComponents: UihtmlComponent[] = [];
  hasUihtmlSession = false;

  private worker: Worker;
  private disposed = false;
  private nextDispatchId = 1;
  private pendingDispatches = new Map<
    number,
    { resolve: () => void; reject: (err: Error) => void }
  >();
  private nextReadId = 1;
  private pendingReads = new Map<
    number,
    { resolve: (content: Uint8Array) => void; reject: (err: Error) => void }
  >();
  private readyWaiter: {
    resolve: () => void;
    reject: (err: Error) => void;
  } | null = null;

  constructor(private options: NumblSessionOptions) {
    const blob = new Blob([workerCode], { type: "text/javascript" });
    const url = URL.createObjectURL(blob);
    this.worker = new Worker(url, { type: "module" });
    // The worker is constructed from the blob synchronously; the URL is only
    // needed until the script has been fetched, which onmessage proves.
    const revokeOnce = () => URL.revokeObjectURL(url);
    this.worker.addEventListener("message", revokeOnce, { once: true });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) =>
      this.handleMessage(e.data);
    this.worker.onerror = e => {
      this.failAll(new Error(`worker error: ${e.message || "unknown"}`));
    };
  }

  start(): Promise<void> {
    const o = this.options;
    this.post({
      type: "boot",
      files: o.files,
      mainFile: o.mainFile,
      mip: o.mip ?? true,
      persistSystem: o.persistSystem ?? true,
      systemInactivityMs: o.systemInactivityMs ?? 24 * 60 * 60 * 1000,
      optimization: o.optimization ?? "1",
      maxIterations: o.maxIterations ?? 1e9,
      displayResults: o.displayResults ?? false,
    });
    return new Promise<void>((resolve, reject) => {
      this.readyWaiter = { resolve, reject };
    });
  }

  writeFile(path: string, content: string | Uint8Array): void {
    this.ensureUsable();
    this.post({ type: "writeFile", path, content });
  }

  readFile(path: string): Promise<Uint8Array> {
    this.ensureUsable();
    const id = this.nextReadId++;
    this.post({ type: "readFile", id, path });
    return new Promise<Uint8Array>((resolve, reject) => {
      this.pendingReads.set(id, { resolve, reject });
    });
  }

  dispatchHtmlEvent(
    compId: string,
    name: string,
    data: unknown
  ): Promise<void> {
    this.ensureUsable();
    const id = this.nextDispatchId++;
    this.post({ type: "dispatch", id, compId, name, data });
    return new Promise<void>((resolve, reject) => {
      this.pendingDispatches.set(id, { resolve, reject });
    });
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.worker.terminate();
    this.failAll(new Error("session disposed"));
  }

  private ensureUsable() {
    if (this.disposed) throw new Error("session disposed");
  }

  private post(msg: ToWorker) {
    this.worker.postMessage(msg);
  }

  private failAll(err: Error) {
    const rw = this.readyWaiter;
    this.readyWaiter = null;
    rw?.reject(err);
    for (const waiter of this.pendingDispatches.values()) waiter.reject(err);
    this.pendingDispatches.clear();
    for (const waiter of this.pendingReads.values()) waiter.reject(err);
    this.pendingReads.clear();
  }

  private handleMessage(msg: FromWorker) {
    switch (msg.type) {
      case "progress":
        this.options.onProgress?.(msg.message);
        break;
      case "output":
        this.options.onOutput?.(msg.text);
        break;
      case "uihtml": {
        const comp = { compId: msg.compId, dataJson: msg.dataJson };
        const i = this.uihtmlComponents.findIndex(
          c => c.compId === comp.compId
        );
        if (i >= 0) this.uihtmlComponents[i] = comp;
        else this.uihtmlComponents.push(comp);
        this.options.onUihtml?.(msg.compId, msg.dataJson);
        break;
      }
      case "ready": {
        this.hasUihtmlSession = msg.hasUihtmlSession;
        this.uihtmlComponents = [...msg.components];
        const rw = this.readyWaiter;
        this.readyWaiter = null;
        rw?.resolve();
        break;
      }
      case "bootError": {
        const rw = this.readyWaiter;
        this.readyWaiter = null;
        rw?.reject(new Error(msg.message));
        break;
      }
      case "htmlSourceEvent":
        this.options.onHtmlSourceEvent?.(msg.compId, msg.name, msg.dataJson);
        break;
      case "dispatchResult": {
        const waiter = this.pendingDispatches.get(msg.id);
        this.pendingDispatches.delete(msg.id);
        if (!waiter) break;
        if (msg.ok) waiter.resolve();
        else waiter.reject(new Error(msg.message ?? "dispatch failed"));
        break;
      }
      case "readFileResult": {
        const waiter = this.pendingReads.get(msg.id);
        this.pendingReads.delete(msg.id);
        if (!waiter) break;
        if (msg.ok && msg.content) waiter.resolve(msg.content);
        else waiter.reject(new Error(msg.message ?? "read failed"));
        break;
      }
    }
  }
}

/**
 * Boot a numbl session in a dedicated worker: restore the persisted /system
 * directory, bootstrap mip, run `mainFile`, and keep the uihtml session live.
 * Resolves once the main script has finished (including any
 * `mip load --install` downloads it performs).
 */
export async function createNumblSession(
  options: NumblSessionOptions
): Promise<NumblSession> {
  const session = new NumblSessionImpl(options);
  try {
    await session.start();
  } catch (err) {
    session.dispose();
    throw err;
  }
  return session;
}
