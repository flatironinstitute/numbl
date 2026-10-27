/**
 * Registers an extended MATLAB language definition with react-syntax-highlighter
 * (highlight.js Light build) that recognizes numbl's full set of builtins as
 * built_in keywords. The default hljs matlab definition only knows a small
 * subset (e.g. it's missing fprintf, xlabel, ylabel, title, etc.).
 */
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import baseMatlab from "react-syntax-highlighter/dist/esm/languages/hljs/matlab";
import { getAllIBuiltinNames } from "../numbl-core/interpreter/builtins/index.js";

let registered = false;

export function registerMatlabHighlighter(): void {
  if (registered) return;
  registered = true;

  const allBuiltins = getAllIBuiltinNames().filter(n => !n.startsWith("__"));

  // Wrap the base matlab language definition and merge our builtins into its
  // built_in keyword list.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const extended = (hljs: any) => {
    const lang = baseMatlab(hljs);
    const existingBuiltIns: string =
      (lang.keywords && lang.keywords.built_in) || "";
    const merged = Array.from(
      new Set([
        ...existingBuiltIns.split(/\s+/).filter(Boolean),
        ...allBuiltins,
      ])
    ).join(" ");
    return {
      ...lang,
      keywords: {
        ...(lang.keywords || {}),
        built_in: merged,
      },
    };
  };

  SyntaxHighlighter.registerLanguage("matlab", extended);
}
