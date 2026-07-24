import gettingStarted from "./getting-started.md?raw";
import language from "./language.md?raw";
import builtins from "./builtins.md?raw";
import plotting from "./plotting.md?raw";
import library from "./library.md?raw";
import deploying from "./deploying.md?raw";
import differences from "./differences.md?raw";
import { docsManifest, type DocMeta } from "./manifest";

export interface DocEntry extends DocMeta {
  content: string;
}

const contentByFile: Record<string, string> = {
  "getting-started.md": gettingStarted,
  "language.md": language,
  "builtins.md": builtins,
  "plotting.md": plotting,
  "library.md": library,
  "deploying.md": deploying,
  "differences.md": differences,
};

export const docs: DocEntry[] = docsManifest.map(meta => {
  const content = contentByFile[meta.file];
  if (content === undefined) {
    throw new Error(`No raw import for doc file ${meta.file}`);
  }
  return { ...meta, content };
});
