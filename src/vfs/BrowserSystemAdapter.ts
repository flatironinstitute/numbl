/**
 * Browser implementation of SystemAdapter using in-memory state.
 */

import type { SystemAdapter } from "../numbl-core/systemAdapter.js";

export class BrowserSystemAdapter implements SystemAdapter {
  private vars = new Map<string, string>();
  private currentDir = "/";

  getEnv(name: string): string | undefined {
    return this.vars.get(name);
  }

  getAllEnv(): Record<string, string> {
    const result: Record<string, string> = {};
    for (const [k, v] of this.vars) {
      result[k] = v;
    }
    return result;
  }

  setEnv(name: string, value: string): void {
    this.vars.set(name, value);
  }

  cwd(): string {
    return this.currentDir;
  }

  chdir(dir: string): void {
    // Simple path resolution for virtual cwd
    if (dir.startsWith("/")) {
      this.currentDir = dir;
    } else {
      this.currentDir = this.currentDir.replace(/\/$/, "") + "/" + dir;
    }
  }

  platform(): string {
    return "linux";
  }

  arch(): string {
    return "x64";
  }
}
