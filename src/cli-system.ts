/**
 * Node.js implementation of SystemAdapter using process.*.
 */

import type { SystemAdapter } from "./numbl-core/systemAdapter.js";

export class NodeSystemAdapter implements SystemAdapter {
  getEnv(name: string): string | undefined {
    return process.env[name];
  }

  getAllEnv(): Record<string, string> {
    const result: Record<string, string> = {};
    for (const [k, v] of Object.entries(process.env)) {
      if (v !== undefined) result[k] = v;
    }
    return result;
  }

  setEnv(name: string, value: string): void {
    process.env[name] = value;
  }

  cwd(): string {
    return process.cwd();
  }

  chdir(dir: string): void {
    process.chdir(dir);
  }

  platform(): string {
    return process.platform;
  }

  arch(): string {
    return process.arch;
  }
}
