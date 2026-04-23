/**
 * Shared content-hash helper for the e1 codegen.
 *
 * 64-bit FNV-1a over UTF-8 code units, returned as 16 hex chars.
 * Deterministic, fully self-contained, and browser-safe (no Node
 * `crypto` dependency). Cryptographic strength isn't needed — the
 * hash is a content-addressed suffix for kernel names and
 * `$h.$kernels[...]` cache keys.
 */

export function fnv1a64Hex(s: string): string {
  let h = 0xcbf29ce484222325n;
  const prime = 0x100000001b3n;
  const mask = 0xffffffffffffffffn;
  for (let i = 0; i < s.length; i++) {
    h = ((h ^ BigInt(s.charCodeAt(i))) * prime) & mask;
  }
  return h.toString(16).padStart(16, "0");
}
