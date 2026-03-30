/**
 * Platform-agnostic system adapter.
 *
 * numbl-core defines this interface but never imports Node.js process.
 * Concrete implementations are injected from the host environment
 * via ExecOptions.system.
 */

export interface SystemAdapter {
  /** Get a single env var value, or undefined if not set. */
  getEnv(name: string): string | undefined;
  /** Get all env vars as key-value pairs. */
  getAllEnv(): Record<string, string>;
  /** Set an env var. */
  setEnv(name: string, value: string): void;

  /** Return the current working directory. */
  cwd(): string;
  /** Change the current working directory. Throws on failure. */
  chdir(dir: string): void;

  /** Return the platform string (e.g. "linux", "darwin", "win32"). */
  platform(): string;
  /** Return the CPU architecture string (e.g. "x64", "arm64"). */
  arch(): string;
}
