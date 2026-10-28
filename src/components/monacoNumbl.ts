import type { Monaco } from "@monaco-editor/react";
import {
  numblLanguageConfig,
  createNumblTokensProvider,
} from "../numblLanguage.js";

/** Register numbl syntax highlighting as language `id` (once per Monaco). */
export function registerNumblLanguage(monaco: Monaco, id: string): void {
  if (monaco.languages.getLanguages().some((l: { id: string }) => l.id === id))
    return;
  monaco.languages.register({ id });
  monaco.languages.setLanguageConfiguration(id, numblLanguageConfig);
  monaco.languages.setMonarchTokensProvider(id, createNumblTokensProvider());
}

/** Define and select the IDE's "numbl-dark" theme. */
export function applyNumblDarkTheme(monaco: Monaco): void {
  monaco.editor.defineTheme("numbl-dark", {
    base: "vs-dark",
    inherit: true,
    rules: [
      { token: "comment", foreground: "6A9955" },
      { token: "keyword", foreground: "C586C0" },
      { token: "number", foreground: "B5CEA8" },
      { token: "string", foreground: "CE9178" },
      { token: "operator", foreground: "D4D4D4" },
    ],
    colors: {},
  });
  monaco.editor.setTheme("numbl-dark");
}
