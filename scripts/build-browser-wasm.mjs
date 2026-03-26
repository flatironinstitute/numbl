#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..");
const targetsDir = join(repoRoot, "browser-wasm", "targets");
const localSourceConfigPath = join(repoRoot, "browser-wasm", "local-sources.json");
const defaultOutputDir = join(repoRoot, "public", "wasm-kernels");

function usage() {
  console.log(`Usage: node scripts/build-browser-wasm.mjs [options] [target...]

Options:
  --list    List available browser Wasm targets
  --all     Attempt every configured target, even if its source tree is missing
  --merge   Merge explicitly built targets into the existing runtime manifest
  --help    Show this help message
`);
}

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}

function relFromRoot(path) {
  return path.startsWith(repoRoot) ? path.slice(repoRoot.length + 1) : path;
}

function toRuntimeWasmPath(path) {
  const relPath = relFromRoot(path).replaceAll("\\", "/");
  return relPath.startsWith("public/") ? relPath.slice("public/".length) : relPath;
}

function expandEnvString(value) {
  return value.replace(/\$\{([A-Z0-9_]+)\}/g, (_match, name) => {
    const resolved = process.env[name];
    if (typeof resolved !== "string" || resolved.length === 0) {
      fail(`missing environment variable for manifest expansion: ${name}`);
    }
    return resolved;
  });
}

function expandManifestValue(value) {
  if (typeof value === "string") {
    return expandEnvString(value);
  }
  if (Array.isArray(value)) {
    return value.map(item => expandManifestValue(item));
  }
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [key, expandManifestValue(item)])
    );
  }
  return value;
}

function loadTargets() {
  if (!existsSync(targetsDir)) {
    fail(`target manifest directory not found: ${targetsDir}`);
  }

  const entries = readdirSync(targetsDir)
    .filter(name => name.endsWith(".json"))
    .sort();

  const targets = [];
  for (const name of entries) {
    const absPath = join(targetsDir, name);
    const raw = readFileSync(absPath, "utf8");
    const target = expandManifestValue(JSON.parse(raw));
    target.manifestFile = absPath;
    targets.push(target);
  }
  return targets;
}

function loadLocalSourceConfig() {
  if (!existsSync(localSourceConfigPath)) {
    return {};
  }
  return JSON.parse(readFileSync(localSourceConfigPath, "utf8"));
}

function listTargets(targets) {
  for (const target of targets) {
    console.log(`${target.name}`);
    if (target.description) {
      console.log(`  ${target.description}`);
    }
    const sourceHint = target.sourceRootEnv
      ? `env:${target.sourceRootEnv} or browser-wasm/local-sources.json`
      : target.sourceRoot
        ? target.sourceRoot
        : target.buildScript
          ? "(build script only)"
          : "(none configured)";
    console.log(`  sourceRoot: ${sourceHint}`);
    console.log(`  output: ${target.output ?? `public/wasm-kernels/${target.name}.wasm`}`);
  }
}

function resolveWithinRoot(baseDir, maybeRelative) {
  return isAbsolute(maybeRelative) ? maybeRelative : resolve(baseDir, maybeRelative);
}

function commandExists(command) {
  const probe = spawnSync(command, ["--version"], {
    stdio: "ignore",
  });
  return probe.status === 0;
}

function ensureCompiler(command) {
  if (!commandExists(command)) {
    fail(
      `${command} was not found on PATH. Install Emscripten and ensure ${command} is available.`
    );
  }
}

function validateTarget(target) {
  if (!target.name || typeof target.name !== "string") {
    fail(`invalid target manifest in ${target.manifestFile}: missing name`);
  }
  if (
    !target.buildScript &&
    (!Array.isArray(target.sources) || target.sources.length === 0)
  ) {
    fail(`invalid target manifest for ${target.name}: missing sources`);
  }
}

function resolveConfiguredSourceRoot(target, localSourceConfig) {
  if (target.sourceRootEnv) {
    const fromEnv = process.env[target.sourceRootEnv];
    if (fromEnv && fromEnv.trim().length > 0) {
      return fromEnv.trim();
    }
  }

  const targetsMap =
    typeof localSourceConfig.targets === "object" &&
    localSourceConfig.targets !== null
      ? localSourceConfig.targets
      : {};
  const localEntry =
    targetsMap[target.name] ??
    (typeof localSourceConfig[target.name] === "string"
      ? { sourceRoot: localSourceConfig[target.name] }
      : localSourceConfig[target.name]);
  if (
    localEntry &&
    typeof localEntry === "object" &&
    typeof localEntry.sourceRoot === "string" &&
    localEntry.sourceRoot.trim().length > 0
  ) {
    return localEntry.sourceRoot.trim();
  }

  if (typeof target.sourceRoot === "string" && target.sourceRoot.length > 0) {
    return target.sourceRoot;
  }

  return null;
}

function buildTarget(target, localSourceConfig) {
  validateTarget(target);
  const configuredSourceRoot = resolveConfiguredSourceRoot(
    target,
    localSourceConfig
  );
  const buildScriptOnly =
    typeof target.buildScript === "string" &&
    target.buildScript.length > 0 &&
    !target.sourceRootEnv &&
    !(typeof target.sourceRoot === "string" && target.sourceRoot.length > 0);
  if (!configuredSourceRoot && !buildScriptOnly) {
    const sourceHint = target.sourceRootEnv
      ? `set ${target.sourceRootEnv} or browser-wasm/local-sources.json`
      : "configure a source root";
    return {
      name: target.name,
      status: "skipped",
      reason: `no source root configured; ${sourceHint}`,
    };
  }

  const sourceRoot = configuredSourceRoot
    ? resolveWithinRoot(repoRoot, configuredSourceRoot)
    : null;
  if (sourceRoot && !existsSync(sourceRoot)) {
    return {
      name: target.name,
      status: "skipped",
      reason: `source tree missing: ${relFromRoot(sourceRoot)}`,
    };
  }

  const outputPath = resolveWithinRoot(
    repoRoot,
    target.output || `public/wasm-kernels/${target.name}.wasm`
  );
  const outputDir = dirname(outputPath);
  mkdirSync(outputDir, { recursive: true });

  if (target.buildScript) {
    const buildScript = resolveWithinRoot(repoRoot, target.buildScript);
    if (!existsSync(buildScript)) {
      return {
        name: target.name,
        status: "skipped",
        reason: `build script missing: ${relFromRoot(buildScript)}`,
      };
    }

    console.log(`Building ${target.name} via ${relFromRoot(buildScript)}`);
    const proc = spawnSync(buildScript, [], {
      cwd: repoRoot,
      stdio: "inherit",
      env: {
        ...process.env,
        NUMBL_BROWSER_WASM_TARGET_NAME: target.name,
        NUMBL_BROWSER_WASM_OUTPUT: outputPath,
        NUMBL_BROWSER_WASM_MANIFEST: target.manifestFile,
        ...(sourceRoot
          ? { NUMBL_BROWSER_WASM_SOURCE_ROOT: sourceRoot }
          : {}),
      },
    });

    if (proc.status !== 0) {
      return {
        name: target.name,
        status: "failed",
        reason: `${relFromRoot(buildScript)} exited with status ${proc.status ?? "unknown"}`,
      };
    }
    if (!existsSync(outputPath)) {
      return {
        name: target.name,
        status: "failed",
        reason: `build script did not create ${relFromRoot(outputPath)}`,
      };
    }
    return {
      name: target.name,
      status: "built",
      wasmPath: toRuntimeWasmPath(outputPath),
      sourceRoot: sourceRoot ? relFromRoot(sourceRoot) : undefined,
      exports: Array.isArray(target.exports) ? target.exports : [],
      enabledByDefault: target.enabledByDefault !== false,
      capabilities:
        target.capabilities && typeof target.capabilities === "object"
          ? target.capabilities
          : undefined,
    };
  }

  const compiler = target.compiler || "em++";
  const sourcePaths = target.sources.map(source =>
    resolveWithinRoot(sourceRoot, source)
  );
  const missingSource = sourcePaths.find(path => !existsSync(path));
  if (missingSource) {
    return {
      name: target.name,
      status: "skipped",
      reason: `source file missing: ${relFromRoot(missingSource)}`,
    };
  }

  ensureCompiler(compiler);

  const includeDirs = Array.isArray(target.includeDirs) ? target.includeDirs : [];
  const includeArgs = includeDirs.flatMap(includeDir => [
    "-I",
    resolveWithinRoot(sourceRoot, includeDir),
  ]);
  const cflags = Array.isArray(target.cflags) ? target.cflags : [];
  const ldflags = Array.isArray(target.ldflags) ? target.ldflags : [];
  const exportsList = Array.isArray(target.exports) ? target.exports : [];

  const args = [
    ...sourcePaths,
    ...includeArgs,
    ...cflags,
    "-O3",
    "-s",
    "STANDALONE_WASM",
    "-s",
    "ALLOW_MEMORY_GROWTH=1",
    "-s",
    "FILESYSTEM=0",
    "-s",
    "ERROR_ON_UNDEFINED_SYMBOLS=0",
    ...(exportsList.length > 0
      ? ["-s", `EXPORTED_FUNCTIONS=${JSON.stringify(exportsList)}`]
      : []),
    ...ldflags,
    "--no-entry",
    "-o",
    outputPath,
  ];

  console.log(`Building ${target.name} -> ${relFromRoot(outputPath)}`);
  const proc = spawnSync(compiler, args, {
    cwd: repoRoot,
    stdio: "inherit",
  });

  if (proc.status !== 0) {
    return {
      name: target.name,
      status: "failed",
      reason: `${compiler} exited with status ${proc.status ?? "unknown"}`,
    };
  }

  return {
    name: target.name,
    status: "built",
    wasmPath: toRuntimeWasmPath(outputPath),
    sourceRoot: sourceRoot ? relFromRoot(sourceRoot) : undefined,
    exports: exportsList,
    enabledByDefault: target.enabledByDefault !== false,
    capabilities:
      target.capabilities && typeof target.capabilities === "object"
        ? target.capabilities
        : undefined,
  };
}

export function mergeRuntimeManifestTargets(
  existingTargets,
  entries,
  preserveExisting = false
) {
  const merged = new Map(
    preserveExisting ? (existingTargets ?? []).map(target => [target.name, target]) : []
  );
  for (const entry of entries) {
    if (entry.status !== "built") continue;
    merged.set(entry.name, {
      name: entry.name,
      wasmPath: entry.wasmPath,
      exports: entry.exports,
      enabledByDefault: entry.enabledByDefault,
      ...(entry.capabilities ? { capabilities: entry.capabilities } : {}),
      ...(entry.sourceRoot ? { sourceRoot: entry.sourceRoot } : {}),
    });
  }
  return [...merged.values()].sort((a, b) => a.name.localeCompare(b.name));
}

function writeRuntimeManifest(entries, existingTargets = [], preserveExisting = false) {
  mkdirSync(defaultOutputDir, { recursive: true });
  const manifestPath = join(defaultOutputDir, "manifest.json");
  const mergedTargets = mergeRuntimeManifestTargets(
    existingTargets,
    entries,
    preserveExisting
  );
  writeFileSync(
    manifestPath,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        targets: mergedTargets,
      },
      null,
      2
    ) + "\n"
  );
  return manifestPath;
}

function readExistingRuntimeManifest() {
  const manifestPath = join(defaultOutputDir, "manifest.json");
  if (!existsSync(manifestPath)) {
    return null;
  }
  const content = readFileSync(manifestPath, "utf8");
  const parsed = JSON.parse(content);
  return {
    path: manifestPath,
    content,
    targets: Array.isArray(parsed?.targets) ? parsed.targets : [],
  };
}

export function main(argv = process.argv.slice(2)) {
  const args = argv;
  if (args.includes("--help")) {
    usage();
    return 0;
  }

  const targets = loadTargets();
  const localSourceConfig = loadLocalSourceConfig();

  if (args.includes("--list")) {
    listTargets(targets);
    return 0;
  }

  const requestedNames = args.filter(arg => !arg.startsWith("--"));
  const forceAll = args.includes("--all");
  const preserveExisting = args.includes("--merge");

  let selectedTargets;
  if (requestedNames.length > 0) {
    selectedTargets = requestedNames.map(name => {
      const target = targets.find(candidate => candidate.name === name);
      if (!target) {
        fail(`unknown browser Wasm target: ${name}`);
      }
      return target;
    });
  } else {
    selectedTargets = targets.filter(target => target.enabledByDefault !== false);
  }

  const results = [];
  const existingManifest = readExistingRuntimeManifest();
  for (const target of selectedTargets) {
    const result = buildTarget(target, localSourceConfig);
    results.push(result);
    if (
      requestedNames.length > 0 &&
      result.status === "skipped" &&
      !forceAll
    ) {
      fail(`${target.name}: ${result.reason}`);
    }
  }

  const builtCount = results.filter(result => result.status === "built").length;
  const skipped = results.filter(result => result.status === "skipped");
  const failed = results.filter(result => result.status === "failed");
  let manifestPath = null;
  let manifestMessage = null;

  if (failed.length === 0) {
    manifestPath = writeRuntimeManifest(
      results,
      existingManifest?.targets ?? [],
      preserveExisting
    );
    manifestMessage = `Wrote ${relFromRoot(manifestPath)}`;
  } else if (existingManifest) {
    manifestPath = existingManifest.path;
    manifestMessage = `Preserved existing ${relFromRoot(manifestPath)}`;
  } else {
    manifestPath = writeRuntimeManifest([]);
    manifestMessage = `Wrote ${relFromRoot(manifestPath)}`;
  }

  for (const result of skipped) {
    console.log(`Skipped ${result.name}: ${result.reason}`);
  }
  for (const result of failed) {
    console.error(`Failed ${result.name}: ${result.reason}`);
  }

  console.log(manifestMessage);

  if (failed.length > 0) {
    return 1;
  }

  if (builtCount === 0) {
    console.log("No browser Wasm targets were built.");
  } else {
    console.log(`Built ${builtCount} browser Wasm target(s).`);
  }

  return 0;
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  process.exitCode = main(process.argv.slice(2));
}
