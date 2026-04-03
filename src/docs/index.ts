import gettingStarted from "./getting-started.md?raw";
import language from "./language.md?raw";
import builtins from "./builtins.md?raw";
import plotting from "./plotting.md?raw";
import library from "./library.md?raw";
import differences from "./differences.md?raw";

export interface DocEntry {
  slug: string;
  title: string;
  content: string;
}

export const docs: DocEntry[] = [
  {
    slug: "getting-started",
    title: "Getting Started",
    content: gettingStarted,
  },
  { slug: "language", title: "Language Features", content: language },
  { slug: "builtins", title: "Built-in Functions", content: builtins },
  { slug: "plotting", title: "Plotting", content: plotting },
  { slug: "library", title: "Library Usage", content: library },
  {
    slug: "differences",
    title: "Differences from MATLAB",
    content: differences,
  },
];
