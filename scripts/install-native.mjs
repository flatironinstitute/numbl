#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { buildNativeDeps } from "./build-native-deps.mjs";
import { resolveNativeAddonConfig } from "./native-addon-config.mjs";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const packageDir = join(__dirname, "..");

function hasFlag(args, flag) {
  return args.includes(flag);
}

function uniqueCandidates(candidates) {
  const seen = new Set();
  const out = [];
  for (const candidate of candidates) {
    const key = `${candidate.command}\0${candidate.args.join("\0")}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(candidate);
  }
  return out;
}

function getBuildCandidates(env, packageDir, existsSyncImpl = existsSync) {
  const localNodeGyp = join(packageDir, "node_modules", "node-gyp", "bin", "node-gyp.js");
  return uniqueCandidates([
    ...(env.npm_config_node_gyp
      ? [
          {
            command: process.execPath,
            args: [env.npm_config_node_gyp, "rebuild"],
            label: "npm-configured node-gyp",
          },
        ]
      : []),
    ...(existsSyncImpl(localNodeGyp)
      ? [
          {
            command: process.execPath,
            args: [localNodeGyp, "rebuild"],
            label: "local node-gyp",
          },
        ]
      : []),
    ...(env.npm_execpath
      ? [
          {
            command: process.execPath,
            args: [env.npm_execpath, "exec", "--yes", "node-gyp", "rebuild"],
            label: "npm exec node-gyp",
          },
        ]
      : []),
    {
      command: "npm",
      args: ["exec", "--yes", "node-gyp", "rebuild"],
      label: "npm exec node-gyp",
    },
    {
      command: "node-gyp",
      args: ["rebuild"],
      label: "node-gyp",
    },
  ]);
}

function runNodeGypBuild(candidates, spawnSyncImpl, spawnOptions) {
  let lastResult = null;
  for (const candidate of candidates) {
    const result = spawnSyncImpl(candidate.command, candidate.args, spawnOptions);
    if (result.error?.code === "ENOENT") {
      lastResult = result;
      continue;
    }
    return {
      ...result,
      candidate,
    };
  }
  return {
    status: null,
    error: lastResult?.error ?? new Error("No usable node-gyp command was found."),
    candidate: candidates[candidates.length - 1] ?? null,
  };
}

function hasExplicitNativeOverrides(env) {
  return [
    "NUMBL_NATIVE_LIBS",
    "NUMBL_NATIVE_INCLUDE_DIRS",
    "NUMBL_NATIVE_DUCC0_INCLUDE_DIRS",
    "NUMBL_NATIVE_CFLAGS",
    "NUMBL_NATIVE_BLAS_PROVIDER",
    "NUMBL_NATIVE_LAPACK_PROVIDER",
    "NUMBL_NATIVE_FFT_PROVIDER",
    "NUMBL_NATIVE_FFT_BACKEND",
    "NUMBL_NATIVE_PROVIDER_HINT",
  ].some(key => String(env[key] ?? "").trim().length > 0);
}

function getHeuristicBuildProfiles(env) {
  if (env.NUMBL_DISABLE_NATIVE_HEURISTICS === "1") {
    return [];
  }

  return [
    {
      label: "system openblas/lapack/fftw3",
      env: {
        ...env,
        NUMBL_NATIVE_LIBS: "-lopenblas -llapack -lfftw3",
        NUMBL_NATIVE_BLAS_PROVIDER: "openblas",
        NUMBL_NATIVE_LAPACK_PROVIDER: "lapack",
        NUMBL_NATIVE_FFT_PROVIDER: "fftw3",
        NUMBL_NATIVE_PROVIDER_HINT: "heuristic-openblas-lapack-fftw3",
      },
    },
    {
      label: "system blis/libflame/fftw3",
      env: {
        ...env,
        NUMBL_NATIVE_LIBS: "-lblis -lflame -lfftw3",
        NUMBL_NATIVE_BLAS_PROVIDER: "blis",
        NUMBL_NATIVE_LAPACK_PROVIDER: "libflame",
        NUMBL_NATIVE_FFT_PROVIDER: "fftw3",
        NUMBL_NATIVE_PROVIDER_HINT: "heuristic-blis-libflame-fftw3",
      },
    },
    {
      label: "system blas/lapack/fftw3",
      env: {
        ...env,
        NUMBL_NATIVE_LIBS: "-lblas -llapack -lfftw3",
        NUMBL_NATIVE_BLAS_PROVIDER: "blas",
        NUMBL_NATIVE_LAPACK_PROVIDER: "lapack",
        NUMBL_NATIVE_FFT_PROVIDER: "fftw3",
        NUMBL_NATIVE_PROVIDER_HINT: "heuristic-blas-lapack-fftw3",
      },
    },
  ];
}

export function runNativeAddonInstall(options = {}) {
  const env = options.env ?? process.env;
  const argv = options.argv ?? process.argv.slice(2);
  const spawnSyncImpl = options.spawnSyncImpl ?? spawnSync;
  const existsSyncImpl = options.existsSyncImpl ?? existsSync;
  const log = options.log ?? console.log;
  const warn = options.warn ?? console.warn;
  const force = hasFlag(argv, "--force") || env.NUMBL_FORCE_NATIVE_BUILD === "1";
  const withDeps =
    hasFlag(argv, "--with-deps") || env.NUMBL_NATIVE_BUILD_FALLBACK === "1";
  const debug = env.NUMBL_DEBUG_NATIVE === "1";
  let envForBuild = { ...env };

  if (
    !force &&
    (env.NUMBL_NO_NATIVE === "1" || env.NUMBL_SKIP_NATIVE_INSTALL === "1")
  ) {
    log("Skipping native addon build: disabled by environment.");
    return { status: "skipped", reason: "disabled" };
  }

  let config = resolveNativeAddonConfig({
    env: envForBuild,
    execFileSyncImpl: options.execFileSyncImpl,
  });

  if (debug) {
    log(`Native provider search order: ${JSON.stringify(config.searchOrder)}`);
    log(`Native provider summary: ${config.providerSummary}`);
  }

  if (!config.canAutoBuild && withDeps) {
    log("Building local BLIS/libFLAME/ducc0 fallback dependencies...");
    const fallback = (options.buildNativeDepsImpl ?? buildNativeDeps)({
      env: envForBuild,
      spawnSyncImpl,
    });
    envForBuild = {
      ...envForBuild,
      ...fallback.env,
    };
    config = resolveNativeAddonConfig({
      env: envForBuild,
      execFileSyncImpl: options.execFileSyncImpl,
    });
    if (debug) {
      log(`Native fallback summary: ${config.providerSummary}`);
    }
  }

  if (!force && !config.canAutoBuild) {
    const canTryHeuristics = !hasExplicitNativeOverrides(envForBuild);
    if (canTryHeuristics) {
      const heuristicProfiles = getHeuristicBuildProfiles(envForBuild);
      for (const profile of heuristicProfiles) {
        const heuristicConfig = resolveNativeAddonConfig({
          env: profile.env,
          execFileSyncImpl: options.execFileSyncImpl,
        });
        log(
          `Attempting native addon build with heuristic providers: ${heuristicConfig.providerSummary}`
        );
        const heuristicResult = runNodeGypBuild(
          getBuildCandidates(env, packageDir, existsSyncImpl),
          spawnSyncImpl,
          {
            cwd: packageDir,
            stdio: "inherit",
            env: profile.env,
          }
        );
        if (heuristicResult.candidate) {
          log(`Using build tool: ${heuristicResult.candidate.label}`);
        }
        if (!heuristicResult.error && heuristicResult.status === 0) {
          log("Native addon built successfully.");
          return {
            status: "built",
            config: heuristicConfig,
            buildTool: heuristicResult.candidate?.label ?? null,
          };
        }
        warn(
          `Warning: heuristic native addon build failed for ${profile.label}.`
        );
      }
    }

    const missing = config.missingCapabilities.join(", ");
    log(`Skipping native addon build: missing ${missing} support.`);
    log(
      "Run `numbl build-addon` after installing system libraries, use `numbl build-addon --with-deps` for the local BLIS/libFLAME/ducc0 fallback, or set NUMBL_NATIVE_LIBS/NUMBL_NATIVE_INCLUDE_DIRS for custom builds."
    );
    return { status: "skipped", reason: "toolchain-missing", config };
  }

  if (force && !config.canAutoBuild) {
    warn(
      "Forcing native addon build even though autodetection is incomplete."
    );
  }

  log("Building native addon...");
  if (config.detectedPackages.blas || config.detectedPackages.lapack || config.detectedPackages.fft) {
    log(
      `Detected native packages: blas=${config.detectedPackages.blas ?? "-"}, lapack=${config.detectedPackages.lapack ?? "-"}, fft=${config.detectedPackages.fft ?? "-"}`
    );
  }
  if (debug || config.providerHint) {
    log(`Selected providers: ${config.providerSummary}`);
  }

  const result = runNodeGypBuild(
    getBuildCandidates(env, packageDir, existsSyncImpl),
    spawnSyncImpl,
    {
      cwd: packageDir,
      stdio: "inherit",
      env: envForBuild,
    }
  );
  if (result.candidate) {
    log(`Using build tool: ${result.candidate.label}`);
  }

  if (result.error || result.status !== 0) {
    warn("Warning: failed to build the native addon. Numbl will use JS fallbacks.");
    return {
      status: "failed",
      code: result.status ?? null,
      error: result.error ?? null,
      config,
      buildTool: result.candidate?.label ?? null,
    };
  }

  log("Native addon built successfully.");
  return { status: "built", config, buildTool: result.candidate?.label ?? null };
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const result = runNativeAddonInstall();
  if (result.status === "failed" && process.argv.includes("--force")) {
    process.exit(result.code ?? 1);
  }
}
