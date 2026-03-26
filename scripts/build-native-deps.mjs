#!/usr/bin/env node

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const repoRoot = resolve(__dirname, "..");
const defaultStageRoot = join(repoRoot, ".cache", "native-deps");
const DEFAULT_BLIS_GIT_REF = "2.0";
const DEFAULT_LIBFLAME_GIT_REF = "5.2.0";
const DEFAULT_DUCC0_GIT_REF = "ducc0";
const NATIVE_SAFE_OPT_FLAGS = [
  "-O3",
  "-DNDEBUG",
  "-fno-fast-math",
  "-fno-math-errno",
  "-ffp-contract=on",
  "-fno-semantic-interposition",
];

function fail(message) {
  throw new Error(message);
}

function ensureDir(path) {
  mkdirSync(path, { recursive: true });
}

function run(command, args, options = {}) {
  const debug = options.debug === true;
  const cwd = options.cwd ?? repoRoot;
  if (debug) {
    process.stderr.write(`[native-deps] ${command} ${args.join(" ")} (cwd=${cwd})\n`);
  }
  const result = (options.spawnSyncImpl ?? spawnSync)(command, args, {
    cwd,
    env: options.env ?? process.env,
    stdio: "inherit",
  });
  if (result.status !== 0) {
    const statusText =
      result.status === null ? "null" : String(result.status);
    fail(
      `${command} ${args.join(" ")} failed (cwd=${cwd}, status=${statusText})`
    );
  }
}

function ensureGitClone(targetDir, url, ref, options = {}) {
  if (!existsSync(join(targetDir, ".git"))) {
    ensureDir(dirname(targetDir));
    const cloneArgs = ["clone", "--depth", "1"];
    if (ref) {
      cloneArgs.push("--branch", ref, "--single-branch");
    }
    cloneArgs.push(url, targetDir);
    run("git", cloneArgs, options);
  }
  if (ref) {
    run("git", ["fetch", "--depth", "1", "origin", ref], {
      ...options,
      cwd: targetDir,
    });
    run("git", ["checkout", "FETCH_HEAD"], {
      ...options,
      cwd: targetDir,
    });
  }
}

function resolveOrCloneSource({
  env,
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
  spawnSyncImpl,
}) {
  const explicitRoot = env[explicitRootEnv];
  const upstreamRoot = explicitRoot
    ? resolve(explicitRoot)
    : existsSync(localProbe)
      ? preferredLocalRoot
      : join(stageRoot, "upstream", cloneDirName);

  if (!existsSync(probePath(upstreamRoot))) {
    ensureGitClone(
      upstreamRoot,
      env[gitUrlEnv] ?? defaultGitUrl,
      env[gitRefEnv] ?? defaultGitRef,
      { spawnSyncImpl }
    );
  }

  return upstreamRoot;
}

function findLibraryDir(prefix, basename) {
  const candidates = [
    join(prefix, "lib", `lib${basename}.a`),
    join(prefix, "lib64", `lib${basename}.a`),
    join(prefix, "lib", `lib${basename}.so`),
    join(prefix, "lib64", `lib${basename}.so`),
    join(prefix, "lib", `lib${basename}.dylib`),
    join(prefix, "lib64", `lib${basename}.dylib`),
  ];
  for (const candidate of candidates) {
    if (existsSync(candidate)) {
      return dirname(candidate);
    }
  }
  return null;
}

function appendFlags(existing, extra) {
  return [existing, ...extra].filter(Boolean).join(" ").trim();
}

function createNativeBuildEnv(env) {
  return {
    ...env,
    CFLAGS: appendFlags(env.CFLAGS, NATIVE_SAFE_OPT_FLAGS),
    CXXFLAGS: appendFlags(env.CXXFLAGS, NATIVE_SAFE_OPT_FLAGS),
  };
}

function buildBlis({
  sourceRoot,
  prefix,
  jobs,
  env,
  spawnSyncImpl,
  configName,
  debug = false,
}) {
  const existingLibDir = findLibraryDir(prefix, "blis");
  if (existingLibDir) return existingLibDir;
  const buildEnv = createNativeBuildEnv(env);

  run(
    join(sourceRoot, "configure"),
    [
      `--prefix=${prefix}`,
      "--disable-shared",
      "--enable-static",
      "--disable-threading",
      configName,
    ],
    {
      cwd: sourceRoot,
      env: buildEnv,
      spawnSyncImpl,
      debug,
    }
  );
  run("make", [`-j${jobs}`], {
    cwd: sourceRoot,
    env: buildEnv,
    spawnSyncImpl,
    debug,
  });
  run("make", ["install"], {
    cwd: sourceRoot,
    env: buildEnv,
    spawnSyncImpl,
    debug,
  });

  const libDir = findLibraryDir(prefix, "blis");
  if (!libDir) {
    fail(`BLIS build did not produce libblis in ${prefix}`);
  }
  return libDir;
}

function buildLibflame({ sourceRoot, prefix, jobs, env, spawnSyncImpl, debug = false }) {
  const existingLibDir = findLibraryDir(prefix, "flame");
  if (existingLibDir) return existingLibDir;

  const buildDir = join(sourceRoot, ".numbl-native-build");
  ensureDir(buildDir);
  const buildEnv = createNativeBuildEnv(env);
  const linkEnv = {
    ...buildEnv,
    LDFLAGS: [buildEnv.LDFLAGS, `-L${prefix}/lib`, `-L${prefix}/lib64`]
      .filter(Boolean)
      .join(" "),
    LIBS: [buildEnv.LIBS, "-lblis"].filter(Boolean).join(" "),
  };

  if (!existsSync(join(buildDir, "Makefile"))) {
    run(
      join(sourceRoot, "configure"),
      [
        `--prefix=${prefix}`,
        "--disable-dynamic-build",
        "--enable-static-build",
        "--disable-autodetect-f77-ldflags",
        "--disable-autodetect-f77-name-mangling",
        "--disable-multithreading",
        "--disable-supermatrix",
        "--disable-gpu",
        "--disable-hip",
        "--disable-vector-intrinsics",
        "--enable-lapack2flame",
        "--enable-legacy-lapack",
        "--disable-external-lapack-for-subproblems",
        "--disable-external-lapack-interfaces",
        "--disable-cblas-interfaces",
        "--disable-builtin-blas",
      ],
      {
        cwd: buildDir,
        env: linkEnv,
        spawnSyncImpl,
        debug,
      }
    );
  }

  run("make", [`-j${jobs}`], {
    cwd: buildDir,
    env: linkEnv,
    spawnSyncImpl,
    debug,
  });
  run("make", ["install"], {
    cwd: buildDir,
    env: linkEnv,
    spawnSyncImpl,
    debug,
  });

  const libDir = findLibraryDir(prefix, "flame");
  if (!libDir) {
    fail(`libFLAME build did not produce libflame in ${prefix}`);
  }
  return libDir;
}

function resolveBlisRoot(stageRoot, env, spawnSyncImpl) {
  const localRoot = "/home/marco/repos/blis";
  return resolveOrCloneSource({
    env,
    explicitRootEnv: "NUMBL_BLIS_UPSTREAM_ROOT",
    preferredLocalRoot: localRoot,
    localProbe: join(localRoot, "configure"),
    stageRoot,
    cloneDirName: "blis",
    gitUrlEnv: "NUMBL_BLIS_GIT_URL",
    gitRefEnv: "NUMBL_BLIS_GIT_REF",
    defaultGitUrl: "https://github.com/flame/blis.git",
    defaultGitRef: DEFAULT_BLIS_GIT_REF,
    probePath: root => join(root, "configure"),
    spawnSyncImpl,
  });
}

function resolveLibflameRoot(stageRoot, env, spawnSyncImpl) {
  const localRoot = "/home/marco/repos/libflame";
  return resolveOrCloneSource({
    env,
    explicitRootEnv: "NUMBL_LIBFLAME_UPSTREAM_ROOT",
    preferredLocalRoot: localRoot,
    localProbe: join(localRoot, "configure"),
    stageRoot,
    cloneDirName: "libflame",
    gitUrlEnv: "NUMBL_LIBFLAME_GIT_URL",
    gitRefEnv: "NUMBL_LIBFLAME_GIT_REF",
    defaultGitUrl: "https://github.com/flame/libflame.git",
    defaultGitRef: DEFAULT_LIBFLAME_GIT_REF,
    probePath: root => join(root, "configure"),
    spawnSyncImpl,
  });
}

function resolveDuccRoot(stageRoot, env, spawnSyncImpl) {
  const localRoot = "/home/marco/repos/ducc";
  return resolveOrCloneSource({
    env,
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
    spawnSyncImpl,
  });
}

export function buildNativeDeps(options = {}) {
  const env = options.env ?? process.env;
  const spawnSyncImpl = options.spawnSyncImpl ?? spawnSync;
  const debug = env.NUMBL_DEBUG_NATIVE === "1";
  const stageRoot = resolve(env.NUMBL_NATIVE_DEPS_ROOT ?? defaultStageRoot);
  const prefix = resolve(
    env.NUMBL_NATIVE_DEPS_PREFIX ?? join(stageRoot, "install")
  );
  const jobs = String(env.NUMBL_NATIVE_DEPS_JOBS ?? "4");
  const blisConfig = String(env.NUMBL_NATIVE_DEPS_BLIS_CONFIG ?? "generic");

  ensureDir(stageRoot);
  ensureDir(prefix);

  const blisRoot = resolveBlisRoot(stageRoot, env, spawnSyncImpl);
  const libflameRoot = resolveLibflameRoot(stageRoot, env, spawnSyncImpl);
  const duccRoot = resolveDuccRoot(stageRoot, env, spawnSyncImpl);

  const blisLibDir = buildBlis({
    sourceRoot: blisRoot,
    prefix,
    jobs,
    env,
    spawnSyncImpl,
    configName: blisConfig,
    debug,
  });
  const flameLibDir = buildLibflame({
    sourceRoot: libflameRoot,
    prefix,
    jobs,
    env,
    spawnSyncImpl,
    debug,
  });
  const duccIncludeDir = join(duccRoot, "src");

  return {
    status: "built",
    prefix,
    sourceRoots: {
      blis: blisRoot,
      libflame: libflameRoot,
      ducc0: duccRoot,
    },
    env: {
      NUMBL_NATIVE_INCLUDE_DIRS: join(prefix, "include"),
      NUMBL_NATIVE_LIBS: [`-L${flameLibDir}`, "-lflame", `-L${blisLibDir}`, "-lblis"].join(
        " "
      ),
      NUMBL_NATIVE_DUCC0_INCLUDE_DIRS: duccIncludeDir,
      NUMBL_NATIVE_FFT_BACKEND: "ducc0",
      NUMBL_NATIVE_PROVIDER_HINT: "local-blis-libflame-ducc0",
      NUMBL_NATIVE_BLAS_PROVIDER: "blis",
      NUMBL_NATIVE_LAPACK_PROVIDER: "libflame",
      NUMBL_NATIVE_FFT_PROVIDER: "ducc0",
      NUMBL_NATIVE_DEPS_PREFIX: prefix,
      NUMBL_NATIVE_DEPS_BLIS_CONFIG: blisConfig,
    },
  };
}

function printEnvLines(env) {
  for (const [key, value] of Object.entries(env)) {
    process.stdout.write(`${key}=${value}\n`);
  }
}

function main(argv) {
  const emitEnv = argv.includes("--env");
  const result = buildNativeDeps();
  if (emitEnv) {
    printEnvLines(result.env);
    return;
  }
  process.stdout.write(JSON.stringify(result, null, 2) + "\n");
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  try {
    main(process.argv.slice(2));
  } catch (error) {
    console.error(
      error instanceof Error ? error.message : String(error)
    );
    process.exit(1);
  }
}
