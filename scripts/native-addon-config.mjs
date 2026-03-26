#!/usr/bin/env node

import { execFileSync } from "node:child_process";
import { basename, delimiter } from "node:path";
import { fileURLToPath } from "node:url";

function splitShellWords(value) {
  const words = [];
  let current = "";
  let quote = null;
  let escaping = false;

  for (const ch of value) {
    if (escaping) {
      current += ch;
      escaping = false;
      continue;
    }
    if (ch === "\\") {
      escaping = true;
      continue;
    }
    if (quote) {
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      continue;
    }
    if (/\s/.test(ch)) {
      if (current) {
        words.push(current);
        current = "";
      }
      continue;
    }
    current += ch;
  }

  if (current) {
    words.push(current);
  }
  return words;
}

function splitListEnv(value) {
  return String(value ?? "")
    .split(/[,\s]+/)
    .map(x => x.trim())
    .filter(Boolean);
}

function splitPathEnv(value) {
  return String(value ?? "")
    .split(delimiter)
    .map(x => x.trim())
    .filter(Boolean);
}

function uniqueTokens(tokens) {
  const seen = new Set();
  const out = [];
  for (const token of tokens) {
    if (!token || seen.has(token)) continue;
    seen.add(token);
    out.push(token);
  }
  return out;
}

function stripIncludeFlags(tokens) {
  return tokens
    .filter(token => token.startsWith("-I"))
    .map(token => token.slice(2))
    .filter(Boolean);
}

function otherCFlags(tokens) {
  return tokens.filter(token => !token.startsWith("-I"));
}

function libraryNameFromToken(token) {
  const raw = String(token ?? "").trim();
  if (!raw) return "";
  if (raw.startsWith("-l")) {
    return raw.slice(2).toLowerCase();
  }
  const trimmed = raw.toLowerCase();
  const base = basename(trimmed);
  const match = base.match(/^lib(.+?)(?:\.(?:so(?:\.\d+)*)|\.a|\.dylib|\.lib)?$/);
  return match ? match[1] : base;
}

function inferCapabilities(libraries) {
  const names = libraries.map(libraryNameFromToken).filter(Boolean);
  const joined = names.join(" ");
  const hasOpenBLAS = names.some(name => name.includes("openblas"));
  const hasBlas =
    hasOpenBLAS ||
    names.some(
      name =>
        name === "blas" ||
        name.startsWith("blas") ||
        name.endsWith("blas") ||
        name.includes("blastrampoline") ||
        name.includes("flexiblas") ||
        name === "blis"
    );
  const hasLapack =
    hasOpenBLAS ||
    names.some(
      name =>
        name === "flame" ||
        name === "libflame" ||
        name.endsWith("flame") ||
        name === "lapack" ||
        name.startsWith("lapack") ||
        name.endsWith("lapack")
    );
  const hasFft = joined.includes("fftw3");
  return {
    usesOpenBLAS: hasOpenBLAS,
    hasBlas,
    hasLapack,
    hasFft,
  };
}

function inferBlasProvider(libraries) {
  const names = libraries.map(libraryNameFromToken).filter(Boolean);
  if (names.some(name => name.includes("openblas"))) return "openblas";
  if (names.some(name => name === "blis")) return "blis";
  if (names.some(name => name.includes("blastrampoline"))) return "blastrampoline";
  if (names.some(name => name.includes("flexiblas"))) return "flexiblas";
  if (
    names.some(
      name =>
        name === "blas" || name.startsWith("blas") || name.endsWith("blas")
    )
  ) {
    return "blas";
  }
  return null;
}

function inferLapackProvider(libraries) {
  const names = libraries.map(libraryNameFromToken).filter(Boolean);
  if (names.some(name => name.includes("openblas"))) return "openblas";
  if (names.some(name => name === "libflame" || name === "flame")) {
    return "libflame";
  }
  if (
    names.some(
      name =>
        name === "lapack" ||
        name.startsWith("lapack") ||
        name.endsWith("lapack")
    )
  ) {
    return "lapack";
  }
  return null;
}

function inferFftProvider(libraries) {
  const names = libraries.map(libraryNameFromToken).filter(Boolean);
  return names.some(name => name.includes("fftw3")) ? "fftw3" : null;
}

function normalizeBlasProvider(value) {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;
  if (normalized.includes("openblas")) return "openblas";
  if (normalized.startsWith("blis")) return "blis";
  if (normalized.includes("blastrampoline")) return "blastrampoline";
  if (normalized.includes("flexiblas")) return "flexiblas";
  if (
    normalized === "blas" ||
    normalized.startsWith("blas") ||
    normalized.endsWith("blas")
  ) {
    return "blas";
  }
  return normalized;
}

function normalizeLapackProvider(value) {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;
  if (normalized.includes("openblas")) return "openblas";
  if (normalized.includes("flame")) return "libflame";
  if (
    normalized === "lapack" ||
    normalized.startsWith("lapack") ||
    normalized.endsWith("lapack")
  ) {
    return "lapack";
  }
  return normalized;
}

function normalizeFftProvider(value) {
  if (!value) return null;
  const normalized = value.trim().toLowerCase();
  if (!normalized) return null;
  if (normalized.includes("fftw3")) return "fftw3";
  if (normalized.includes("ducc0") || normalized === "ducc") return "ducc0";
  return normalized;
}

function stringDefine(name, value) {
  return `${name}=\\\"${String(value)}\\\"`;
}

function pkgConfig(args, execFileSyncImpl = execFileSync) {
  try {
    return execFileSyncImpl("pkg-config", args, {
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return null;
  }
}

function resolvePkgGroup(packages, execFileSyncImpl) {
  for (const packageName of packages) {
    const libsRaw = pkgConfig([packageName, "--libs"], execFileSyncImpl);
    if (libsRaw === null) continue;
    const cflagsRaw =
      pkgConfig([packageName, "--cflags"], execFileSyncImpl) ?? "";
    return {
      packageName,
      libs: splitShellWords(libsRaw),
      cflags: splitShellWords(cflagsRaw),
    };
  }
  return null;
}

export function resolveNativeAddonConfig(options = {}) {
  const env = options.env ?? process.env;
  const execFileSyncImpl = options.execFileSyncImpl ?? execFileSync;

  const manualLibs = splitShellWords(env.NUMBL_NATIVE_LIBS ?? "");
  const manualIncludeDirs = splitPathEnv(env.NUMBL_NATIVE_INCLUDE_DIRS ?? "");
  const manualDuccIncludeDirs = splitPathEnv(
    env.NUMBL_NATIVE_DUCC0_INCLUDE_DIRS ?? ""
  );
  const manualCflags = splitShellWords(env.NUMBL_NATIVE_CFLAGS ?? "");
  const explicitFftBackend = String(env.NUMBL_NATIVE_FFT_BACKEND ?? "")
    .trim()
    .toLowerCase();
  const explicitBlasProvider =
    normalizeBlasProvider(String(env.NUMBL_NATIVE_BLAS_PROVIDER ?? ""));
  const explicitLapackProvider =
    normalizeLapackProvider(String(env.NUMBL_NATIVE_LAPACK_PROVIDER ?? ""));
  const explicitFftProvider =
    normalizeFftProvider(String(env.NUMBL_NATIVE_FFT_PROVIDER ?? ""));
  const providerHint =
    String(env.NUMBL_NATIVE_PROVIDER_HINT ?? "").trim() || null;

  const blasPackages = splitListEnv(
    env.NUMBL_NATIVE_BLAS_PKGS ??
      "blis blis-serial blis-pthread blis-openmp openblas flexiblas blas"
  );
  const lapackPackages = splitListEnv(
    env.NUMBL_NATIVE_LAPACK_PKGS ?? "libflame flame lapack openblas"
  );
  const fftPackages = splitListEnv(env.NUMBL_NATIVE_FFT_PKGS ?? "fftw3");

  const detectedBlas = manualLibs.length
    ? null
    : resolvePkgGroup(blasPackages, execFileSyncImpl);
  const detectedLapack = manualLibs.length
    ? null
    : resolvePkgGroup(lapackPackages, execFileSyncImpl);
  const detectedFft =
    manualLibs.length || explicitFftBackend === "ducc0" || manualDuccIncludeDirs.length > 0
    ? null
    : resolvePkgGroup(fftPackages, execFileSyncImpl);

  const detectedMathLibs = uniqueTokens([
    ...(detectedBlas?.libs ?? []),
    ...(detectedLapack?.libs ?? []),
  ]);
  const detectedMathCaps = inferCapabilities(detectedMathLibs);
  const needLapackFallback =
    !manualLibs.length &&
    (detectedBlas !== null || detectedLapack !== null) &&
    !detectedMathCaps.hasLapack;
  const needBlasFallback =
    !manualLibs.length &&
    (detectedBlas !== null || detectedLapack !== null) &&
    !detectedMathCaps.hasBlas;
  const libraries = uniqueTokens(
    manualLibs.length
      ? manualLibs
      : [
          ...detectedMathLibs,
          ...(needLapackFallback ? ["-llapack"] : []),
          ...(needBlasFallback ? ["-lblas"] : []),
          ...(detectedFft?.libs ?? []),
        ]
  );

  const includeDirs = uniqueTokens([
    ...manualIncludeDirs,
    ...manualDuccIncludeDirs,
    ...stripIncludeFlags(detectedBlas?.cflags ?? []),
    ...stripIncludeFlags(detectedLapack?.cflags ?? []),
    ...stripIncludeFlags(detectedFft?.cflags ?? []),
  ]);

  const cflagsCc = uniqueTokens([
    ...(env.NUMBL_DISABLE_MARCH_NATIVE === "1" ? [] : ["-march=native"]),
    ...otherCFlags(detectedBlas?.cflags ?? []),
    ...otherCFlags(detectedLapack?.cflags ?? []),
    ...otherCFlags(detectedFft?.cflags ?? []),
    ...manualCflags,
  ]);

  const resolvedCaps = inferCapabilities(libraries);
  const selectedProviders = {
    blas:
      explicitBlasProvider ??
      normalizeBlasProvider(detectedBlas?.packageName ?? "") ??
      inferBlasProvider(libraries),
    lapack:
      explicitLapackProvider ??
      normalizeLapackProvider(detectedLapack?.packageName ?? "") ??
      inferLapackProvider(libraries),
    fft:
      explicitFftProvider ??
      (explicitFftBackend === "ducc0" || manualDuccIncludeDirs.length > 0
        ? "ducc0"
        : explicitFftBackend === "fftw3"
          ? "fftw3"
          : normalizeFftProvider(detectedFft?.packageName ?? "") ??
            inferFftProvider(libraries)),
  };
  const usesOpenBLAS =
    resolvedCaps.usesOpenBLAS ||
    selectedProviders.blas === "openblas" ||
    selectedProviders.lapack === "openblas";
  const hasDucc0 = selectedProviders.fft === "ducc0";

  const defines = uniqueTokens([
    ...(usesOpenBLAS ? ["NUMBL_USE_OPENBLAS=1"] : []),
    ...(selectedProviders.fft === "fftw3" ? ["NUMBL_USE_FFTW=1"] : []),
    ...(hasDucc0 ? ["NUMBL_USE_DUCC0=1"] : []),
    stringDefine("NUMBL_BLAS_PROVIDER", selectedProviders.blas ?? "unknown"),
    stringDefine("NUMBL_LAPACK_PROVIDER", selectedProviders.lapack ?? "unknown"),
    stringDefine("NUMBL_FFT_PROVIDER", selectedProviders.fft ?? "unknown"),
  ]);

  const hasMathEvidence =
    manualLibs.length > 0
      ? resolvedCaps.hasBlas && resolvedCaps.hasLapack
      : detectedBlas !== null || detectedLapack !== null;
  const hasFftEvidence =
    manualLibs.length > 0
      ? resolvedCaps.hasFft || hasDucc0
      : detectedFft !== null || hasDucc0;
  const missingCapabilities = [];
  if (!resolvedCaps.hasBlas) missingCapabilities.push("BLAS");
  if (!resolvedCaps.hasLapack) missingCapabilities.push("LAPACK");
  if (!(resolvedCaps.hasFft || hasDucc0)) {
    missingCapabilities.push("FFT");
  }
  const searchOrder = {
    blas: blasPackages,
    lapack: lapackPackages,
    fft: explicitFftBackend === "ducc0" ? ["ducc0", ...fftPackages] : fftPackages,
  };
  const providerSummary = `blas=${selectedProviders.blas ?? "-"}, lapack=${selectedProviders.lapack ?? "-"}, fft=${selectedProviders.fft ?? "-"}`;

  return {
    libraries,
    includeDirs,
    cflagsCc,
    defines,
    canAutoBuild:
      hasMathEvidence &&
      hasFftEvidence &&
      resolvedCaps.hasBlas &&
      resolvedCaps.hasLapack &&
      (resolvedCaps.hasFft || hasDucc0),
    detectedPackages: {
      blas: detectedBlas?.packageName ?? null,
      lapack: detectedLapack?.packageName ?? null,
      fft: detectedFft?.packageName ?? null,
    },
    selectedProviders,
    searchOrder,
    providerHint,
    providerSummary,
    resolvedCapabilities: {
      blas: resolvedCaps.hasBlas,
      lapack: resolvedCaps.hasLapack,
      fft: resolvedCaps.hasFft || hasDucc0,
      fftBackend: selectedProviders.fft,
      usesOpenBLAS,
    },
    missingCapabilities,
  };
}

function printTokens(tokens) {
  if (tokens.length > 0) {
    process.stdout.write(tokens.join("\n") + "\n");
  }
}

function main(argv) {
  const command = argv[2] ?? "summary";
  const config = resolveNativeAddonConfig();

  switch (command) {
    case "libraries":
      printTokens(config.libraries);
      break;
    case "include-dirs":
      printTokens(config.includeDirs);
      break;
    case "cflags-cc":
      printTokens(config.cflagsCc);
      break;
    case "defines":
      printTokens(config.defines);
      break;
    case "summary":
      process.stdout.write(JSON.stringify(config, null, 2) + "\n");
      break;
    default:
      console.error(
        `Unknown command: ${command}. Use one of: libraries, include-dirs, cflags-cc, defines, summary.`
      );
      process.exit(1);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main(process.argv);
}
