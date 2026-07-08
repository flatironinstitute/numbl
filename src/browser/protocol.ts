/** Messages between the NumblSession host and its worker. */

export interface BootFile {
  path: string;
  content: string | Uint8Array;
}

export interface BootMessage {
  type: "boot";
  files: BootFile[];
  mainFile: string;
  mip: boolean;
  persistSystem: boolean;
  systemInactivityMs: number;
  optimization: "0" | "1";
  maxIterations: number;
  displayResults: boolean;
}

export type ToWorker =
  | BootMessage
  | { type: "writeFile"; path: string; content: string | Uint8Array }
  | {
      type: "dispatch";
      id: number;
      compId: string;
      name: string;
      data: unknown;
    };

export interface UihtmlComponent {
  compId: string;
  dataJson: string;
}

export type FromWorker =
  | { type: "progress"; message: string }
  | { type: "output"; text: string }
  | { type: "uihtml"; compId: string; dataJson: string }
  | {
      type: "ready";
      hasUihtmlSession: boolean;
      components: UihtmlComponent[];
    }
  | { type: "bootError"; message: string }
  | { type: "htmlSourceEvent"; compId: string; name: string; dataJson: string }
  | { type: "dispatchResult"; id: number; ok: boolean; message?: string };
