/**
 * Specialization key computation.
 *
 * Shared between the lowering and codegen layers for cache keying
 * of specialized function versions by (name, argTypes).
 */

import { type ItemType, typeToString } from "../lowering/itemTypes.js";

/**
 * Compute a specialization cache key as a deterministic JSON string.
 * Uses typeToString to capture full type info (isComplex, isLogical,
 * className, knownFields, etc.) so that distinct types get distinct keys.
 */
export function computeSpecKey(name: string, argTypes: ItemType[]): string {
  const args = argTypes.map(t => typeToString(t));
  return JSON.stringify({ name, args });
}

/**
 * Compute a short hash string from argument types for use in JS function IDs.
 * Uses FNV-1a to produce an 8-character hex string.
 */
export function hashForJsId(argTypes: ItemType[]): string {
  const args = argTypes.map(t => typeToString(t));
  const input = JSON.stringify(args);
  let hash = 0x811c9dc5; // FNV offset basis
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193); // FNV prime
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}
