#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..");
const defaultStageRoot = join(repoRoot, ".cache", "browser-wasm");
const DEFAULT_DUCC0_GIT_REF = "ducc0";
const DEFAULT_OPENBLAS_GIT_REF = "v0.3.32";
const DEFAULT_LIBFLAME_GIT_REF = "5.2.0";

function fail(message) {
  console.error(`Error: ${message}`);
  process.exit(1);
}

function ensureDir(path) {
  mkdirSync(path, { recursive: true });
}

function run(command, args, cwd) {
  const result = spawnSync(command, args, {
    cwd,
    stdio: "inherit",
  });
  if (result.status !== 0) {
    fail(`${command} ${args.join(" ")} failed`);
  }
}

function ensureGitClone(targetDir, url, ref) {
  if (!existsSync(join(targetDir, ".git"))) {
    ensureDir(dirname(targetDir));
    const cloneArgs = ["clone", "--depth", "1"];
    if (ref) {
      cloneArgs.push("--branch", ref, "--single-branch");
    }
    cloneArgs.push(url, targetDir);
    run("git", cloneArgs, repoRoot);
  }
  if (ref) {
    run("git", ["fetch", "--depth", "1", "origin", ref], targetDir);
    run("git", ["checkout", "FETCH_HEAD"], targetDir);
  }
}

function resolveOrCloneSource({
  explicitRootEnv,
  preferredLocalRoot,
  localProbe,
  stageRoot,
  cloneDirName,
  gitUrlEnv,
  gitRefEnv,
  defaultGitUrl,
  defaultGitRef,
  probePath,
}) {
  const explicitRoot = process.env[explicitRootEnv];
  const upstreamRoot = explicitRoot
    ? resolve(explicitRoot)
    : existsSync(localProbe)
      ? preferredLocalRoot
      : join(stageRoot, "upstream", cloneDirName);

  if (!existsSync(probePath(upstreamRoot))) {
    ensureGitClone(
      upstreamRoot,
      process.env[gitUrlEnv] ?? defaultGitUrl,
      process.env[gitRefEnv] ?? defaultGitRef
    );
  }

  return upstreamRoot;
}

function resolveDuccRoot(stageRoot) {
  const localRoot = "/home/marco/repos/ducc";
  return resolveOrCloneSource({
    explicitRootEnv: "NUMBL_DUCC0_UPSTREAM_ROOT",
    preferredLocalRoot: localRoot,
    localProbe: join(localRoot, "src", "ducc0", "fft", "fft.h"),
    stageRoot,
    cloneDirName: "ducc",
    gitUrlEnv: "NUMBL_DUCC0_GIT_URL",
    gitRefEnv: "NUMBL_DUCC0_GIT_REF",
    defaultGitUrl: "https://github.com/DiamonDinoia/ducc.git",
    defaultGitRef: DEFAULT_DUCC0_GIT_REF,
    probePath: root => join(root, "src", "ducc0", "fft", "fft.h"),
  });
}

function resolveOpenBLASRoot(stageRoot) {
  const tempClone = "/tmp/numbl-wasm-sources/OpenBLAS";
  const localRepo = "/home/marco/repos/OpenBLAS";
  const preferredLocalRoot = existsSync(join(tempClone, "cblas.h"))
    ? tempClone
    : localRepo;

  return resolveOrCloneSource({
    explicitRootEnv: "NUMBL_OPENBLAS_UPSTREAM_ROOT",
    preferredLocalRoot,
    localProbe: join(preferredLocalRoot, "cblas.h"),
    stageRoot,
    cloneDirName: "OpenBLAS",
    gitUrlEnv: "NUMBL_OPENBLAS_GIT_URL",
    gitRefEnv: "NUMBL_OPENBLAS_GIT_REF",
    defaultGitUrl: "https://github.com/OpenMathLib/OpenBLAS.git",
    defaultGitRef: DEFAULT_OPENBLAS_GIT_REF,
    probePath: root => join(root, "cblas.h"),
  });
}

function resolveLibflameRoot(stageRoot) {
  const tempClone = "/tmp/numbl-wasm-sources/libflame";
  const localRepo = "/home/marco/repos/libflame";
  const preferredLocalRoot = existsSync(join(tempClone, "configure"))
    ? tempClone
    : localRepo;

  return resolveOrCloneSource({
    explicitRootEnv: "NUMBL_LIBFLAME_UPSTREAM_ROOT",
    preferredLocalRoot,
    localProbe: join(preferredLocalRoot, "configure"),
    stageRoot,
    cloneDirName: "libflame",
    gitUrlEnv: "NUMBL_LIBFLAME_GIT_URL",
    gitRefEnv: "NUMBL_LIBFLAME_GIT_REF",
    defaultGitUrl: "https://github.com/flame/libflame.git",
    defaultGitRef: DEFAULT_LIBFLAME_GIT_REF,
    probePath: root => join(root, "configure"),
  });
}

function usage() {
  console.log(`Usage: node scripts/prepare-browser-wasm-sources.mjs <target...>

Currently supported:
  ducc0-fft
  flame-blas-lapack
  blas-lapack
`);
}

const targets = process.argv.slice(2);
if (targets.includes("--help") || targets.length === 0) {
  usage();
  process.exit(targets.length === 0 ? 1 : 0);
}

const stageRoot = process.env.NUMBL_BROWSER_WASM_STAGE_ROOT
  ? resolve(process.env.NUMBL_BROWSER_WASM_STAGE_ROOT)
  : defaultStageRoot;
ensureDir(stageRoot);

for (const target of targets) {
  switch (target) {
    case "ducc0-fft": {
      const sourceRoot = resolveDuccRoot(stageRoot);
      process.stdout.write(`NUMBL_DUCC0_FFT_SRC_ROOT=${sourceRoot}\n`);
      break;
    }
    case "flame-blas-lapack": {
      const sourceRoot = resolveLibflameRoot(stageRoot);
      process.stdout.write(`NUMBL_FLAME_BLAS_LAPACK_SRC_ROOT=${sourceRoot}\n`);
      break;
    }
    case "blas-lapack": {
      const sourceRoot = resolveOpenBLASRoot(stageRoot);
      process.stdout.write(`NUMBL_BLAS_LAPACK_SRC_ROOT=${sourceRoot}\n`);
      break;
    }
    default:
      fail(`unsupported browser Wasm source target: ${target}`);
  }
}
