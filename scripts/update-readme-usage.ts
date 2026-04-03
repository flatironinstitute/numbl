#!/usr/bin/env tsx
/**
 * Updates the CLI help section in README.md from `numbl --help` output.
 *
 * Usage:
 *   npx tsx scripts/update-readme-usage.ts          # update README.md
 *   npx tsx scripts/update-readme-usage.ts --check   # exit 1 if out of date
 */
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const readmePath = resolve(root, "README.md");
const cliPath = resolve(root, "src/cli.ts");

const BEGIN = "<!-- BEGIN CLI HELP -->";
const END = "<!-- END CLI HELP -->";

// Get current help output
const helpText = execFileSync("npx", ["tsx", cliPath, "--help"], {
  encoding: "utf-8",
  cwd: root,
}).trim();

// Replace "npx tsx src/cli.ts" with "numbl" in the output (in case cli prints its own path)
const normalised = helpText.replace(/npx tsx src\/cli\.ts/g, "numbl");

const newBlock = `${BEGIN}\n\`\`\`\n${normalised}\n\`\`\`\n${END}`;

const readme = readFileSync(readmePath, "utf-8");
const re = new RegExp(
  `${BEGIN.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}[\\s\\S]*?${END.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}`
);

if (!re.test(readme)) {
  console.log("README.md does not contain CLI help markers — skipping.");
  process.exit(0);
}

const updated = readme.replace(re, newBlock);

if (process.argv.includes("--check")) {
  if (updated !== readme) {
    console.error(
      "README.md CLI help section is out of date. Run: npm run update-readme"
    );
    process.exit(1);
  }
  console.log("README.md CLI help section is up to date.");
  process.exit(0);
}

writeFileSync(readmePath, updated);
console.log("README.md CLI help section updated.");
