/**
 * Command-line option parsing for the numbl CLI (built on node:util parseArgs).
 */

import { parseArgs, type ParseArgsConfig } from "node:util";
import { delimiter, resolve } from "path";
import {
  OPT_LEVELS,
  isOptLevel,
  type OptLevel,
} from "./numbl-core/executors/plugins.js";

type OptionsConfig = NonNullable<ParseArgsConfig["options"]>;

/** Strictly parse a command's arguments; print the error and exit on a bad
 *  or unknown option. */
export function parseCommandArgs<const O extends OptionsConfig>(
  args: string[],
  options: O
) {
  try {
    return parseArgs({ args, options, allowPositionals: true, tokens: true });
  } catch (e) {
    const err = e as Error & { code?: string };
    const unknown =
      err.code === "ERR_PARSE_ARGS_UNKNOWN_OPTION" &&
      /'([^']+)'/.exec(err.message);
    console.error(
      unknown ? `Unknown option: ${unknown[1]}` : `Error: ${err.message}`
    );
    process.exit(1);
  }
}

/** Parse a numeric option value, exiting with `message` if it is invalid. */
export function parseNumberOption(
  value: string,
  parse: (s: string) => number,
  message: string
): number {
  const n = parse(value);
  if (isNaN(n)) {
    console.error(message);
    process.exit(1);
  }
  return n;
}

/** Options shared by run, eval, run-tests and the REPL. */
const RUN_OPTIONS = {
  "dump-js": { type: "string" },
  "dump-c": { type: "string" },
  "dump-ast": { type: "boolean" },
  "fast-math": { type: "boolean" },
  "no-fast-math": { type: "boolean" },
  "no-native": { type: "boolean" },
  verbose: { type: "boolean" },
  stream: { type: "boolean" },
  plot: { type: "boolean" },
  "plot-port": { type: "string" },
  path: { type: "string", multiple: true },
  profile: { type: "string" },
  opt: { type: "string" },
} as const;

export interface RunOptions {
  dumpJs: string | undefined;
  dumpC: string | undefined;
  dumpAst: boolean;
  verbose: boolean;
  stream: boolean;
  plot: boolean;
  plotPort: number | undefined;
  extraPaths: string[];
  positional: string[];
  profileOutput: string | undefined;
  optimization: OptLevel;
  fastMath: boolean;
  noNative: boolean;
}

/** Directories from the NUMBL_PATH environment variable (platform path
 *  separator, leading `~` expanded), resolved against the cwd. */
function numblPathDirs(): string[] {
  const envPath = process.env.NUMBL_PATH;
  if (!envPath) return [];
  const home = process.env.HOME || process.env.USERPROFILE || "";
  const dirs: string[] = [];
  for (const p of envPath.split(delimiter)) {
    let trimmed = p.trim();
    if (!trimmed) continue;
    if (trimmed.startsWith("~/") || trimmed === "~") {
      trimmed = home + trimmed.slice(1);
    }
    dirs.push(resolve(process.cwd(), trimmed));
  }
  return dirs;
}

export function parseRunOptions(args: string[]): RunOptions {
  const { values, positionals, tokens } = parseCommandArgs(args, RUN_OPTIONS);
  const abs = (p: string | undefined) =>
    p === undefined ? undefined : resolve(process.cwd(), p);

  let optimization: OptLevel = "1";
  if (values.opt !== undefined) {
    if (!isOptLevel(values.opt)) {
      console.error(
        `Error: --opt must be one of ${OPT_LEVELS.join(", ")} (got ${values.opt}).`
      );
      process.exit(1);
    }
    optimization = values.opt;
  }

  const plotPort =
    values["plot-port"] === undefined
      ? undefined
      : parseNumberOption(
          values["plot-port"],
          s => parseInt(s, 10),
          "Error: --plot-port requires a valid port number"
        );

  // Default OFF: the native addon's -ffast-math reorders reductions and
  // vectorizes transcendentals, so results drift from the JIT kernels and
  // diverge across --opt levels. `--fast-math` opts in (with a matching
  // `build-addon --fast-math`); `--no-fast-math` is accepted for back-compat.
  // The last of the two wins.
  const fastMathFlag = tokens
    .filter(
      t =>
        t.kind === "option" &&
        (t.name === "fast-math" || t.name === "no-fast-math")
    )
    .at(-1);

  return {
    dumpJs: abs(values["dump-js"]),
    dumpC: abs(values["dump-c"]),
    dumpAst: values["dump-ast"] ?? false,
    verbose: values.verbose ?? false,
    stream: values.stream ?? false,
    plot: (values.plot ?? false) || plotPort !== undefined,
    plotPort,
    extraPaths: [
      ...numblPathDirs(),
      ...(values.path ?? []).map(p => resolve(process.cwd(), p)),
    ],
    positional: positionals,
    profileOutput: abs(values.profile),
    optimization,
    fastMath:
      fastMathFlag?.kind === "option" && fastMathFlag.name === "fast-math",
    noNative: values["no-native"] ?? false,
  };
}
