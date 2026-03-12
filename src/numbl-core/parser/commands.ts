import { Expr } from "./types.js";

export type CommandArgKind =
  | {
      type: "Keyword";
      allowed?: string[];
      optional: boolean;
      multiKeyword?: boolean;
    }
  | { type: "Any" };

export interface CommandVerb {
  name: string;
  argKind: CommandArgKind;
}

export const COMMAND_VERBS: CommandVerb[] = [
  {
    name: "hold",
    argKind: {
      type: "Keyword",
      allowed: ["on", "off", "all", "reset"],
      optional: false,
    },
  },
  {
    name: "pause",
    argKind: {
      type: "Keyword",
      allowed: ["on", "off"],
      optional: true,
    },
  },
  {
    name: "warning",
    argKind: {
      type: "Keyword",
      allowed: ["on", "off"],
      optional: true,
    },
  },
  {
    name: "grid",
    argKind: { type: "Keyword", allowed: ["on", "off"], optional: false },
  },
  {
    name: "box",
    argKind: { type: "Keyword", allowed: ["on", "off"], optional: false },
  },
  {
    name: "uiwait",
    argKind: {
      type: "Keyword",
      allowed: undefined,
      optional: true,
    },
  },
  {
    name: "axis",
    argKind: {
      type: "Keyword",
      allowed: [
        "auto",
        "manual",
        "tight",
        "equal",
        "ij",
        "xy",
        "off",
        "square",
        "vis3d",
      ],
      optional: false,
      multiKeyword: true,
    },
  },
  { name: "clear", argKind: { type: "Any" } },
  { name: "clf", argKind: { type: "Any" } },
  {
    name: "shading",
    argKind: {
      type: "Keyword",
      allowed: ["flat", "interp", "faceted"],
      optional: false,
    },
  },
  {
    name: "colormap",
    argKind: {
      type: "Keyword",
      allowed: [
        "parula",
        "jet",
        "hsv",
        "hot",
        "cool",
        "spring",
        "summer",
        "autumn",
        "winter",
        "gray",
        "bone",
        "copper",
        "pink",
        "default",
      ],
      optional: false,
    },
  },
  {
    name: "lighting",
    argKind: {
      type: "Keyword",
      // allow any string here since users can define their own lighting styles
      allowed: undefined,
      optional: false,
    },
  },
  {
    name: "material",
    argKind: {
      type: "Keyword",
      allowed: undefined,
      optional: false,
    },
  },
  {
    name: "colorbar",
    argKind: { type: "Keyword", allowed: ["on", "off"], optional: true },
  },
  {
    name: "format",
    argKind: {
      type: "Keyword",
      allowed: undefined,
      optional: false,
    },
  },
  { name: "figure", argKind: { type: "Any" } },
  { name: "subplot", argKind: { type: "Any" } },
  { name: "clf", argKind: { type: "Any" } },
  { name: "cla", argKind: { type: "Any" } },
  {
    name: "close",
    argKind: { type: "Keyword", allowed: ["all"], optional: true },
  },
];

export function extractKeyword(expr: Expr): string | null {
  if (expr.type === "Ident") return expr.name;
  if (expr.type === "Char") {
    return expr.value.replace(/^'|'$/g, "");
  }
  if (expr.type === "String") {
    return expr.value.replace(/^"|"$/g, "");
  }
  return null;
}
