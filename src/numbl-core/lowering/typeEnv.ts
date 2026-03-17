/**
 * Type environment: maps variable IDs to their inferred types.
 * Owned by LoweringContext; decoupled from IRVariable so that
 * types can later vary by program point (flow-dependent typing).
 */

import { type VarId } from "./varId.js";
import { type ItemType, IType } from "./itemTypes.js";

export class TypeEnv {
  private types = new Map<string, ItemType>();

  get(id: VarId): ItemType | undefined {
    return this.types.get(id.id);
  }

  set(id: VarId, ty: ItemType | undefined): void {
    if (ty !== undefined) {
      this.types.set(id.id, ty);
    }
  }

  /** Unify the current type for `id` with `ty` and store the result. */
  unify(id: VarId, ty: ItemType): void {
    const current = this.types.get(id.id);
    const unified = IType.unify(current, ty);
    if (unified !== undefined) {
      this.types.set(id.id, unified);
    }
  }

  /** Read type, returning Unknown if unset. */
  getOrUnknown(id: VarId): ItemType {
    return this.types.get(id.id) ?? IType.Unknown;
  }
}
