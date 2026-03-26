import { describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { delimiter } from "node:path";
// @ts-expect-error tests import the Node build helper directly
import { resolveNativeAddonConfig } from "../../scripts/native-addon-config.mjs";
// @ts-expect-error tests import the Node build helper directly
import { runNativeAddonInstall } from "../../scripts/install-native.mjs";

describe("package manifest", () => {
  it("publishes native and browser wasm build support files", () => {
    const pkg = JSON.parse(readFileSync("package.json", "utf8"));

    expect(pkg.files).toEqual(
      expect.arrayContaining([
        "browser-wasm/",
        "docs/browser-wasm.md",
        "docs/native-addon.md",
        "scripts/build-native-deps.mjs",
        "scripts/build-browser-wasm.mjs",
        "scripts/install-native.mjs",
        "scripts/native-addon-config.mjs",
        "scripts/prepare-browser-wasm-sources.mjs",
      ])
    );
  });
});

describe("native addon config", () => {
  it("honors explicit linker, include, and compiler overrides", () => {
    const env = {
      NUMBL_NATIVE_LIBS: "-L/opt/math/lib -lopenblas -lfftw3",
      NUMBL_NATIVE_INCLUDE_DIRS: ["/opt/math/include", "/opt/fftw/include"].join(
        delimiter
      ),
      NUMBL_NATIVE_CFLAGS: "-mfma -funroll-loops",
    };

    const config = resolveNativeAddonConfig({
      env,
      execFileSyncImpl: vi.fn(),
    });

    expect(config.libraries).toEqual([
      "-L/opt/math/lib",
      "-lopenblas",
      "-lfftw3",
    ]);
    expect(config.linkerFlags).toEqual(["-Wl,-rpath,/opt/math/lib"]);
    expect(config.includeDirs).toEqual([
      "/opt/math/include",
      "/opt/fftw/include",
    ]);
    expect(config.cflagsCc).toEqual([
      "-march=native",
      "-mfma",
      "-funroll-loops",
    ]);
    expect(config.defines).toEqual([
      "NUMBL_USE_OPENBLAS=1",
      "NUMBL_USE_FFTW=1",
      'NUMBL_BLAS_PROVIDER=\\"openblas\\"',
      'NUMBL_LAPACK_PROVIDER=\\"openblas\\"',
      'NUMBL_FFT_PROVIDER=\\"fftw3\\"',
    ]);
    expect(config.selectedProviders).toEqual({
      blas: "openblas",
      lapack: "openblas",
      fft: "fftw3",
    });
    expect(config.resolvedCapabilities).toEqual({
      blas: true,
      lapack: true,
      fft: true,
      fftBackend: "fftw3",
      usesOpenBLAS: true,
    });
    expect(config.canAutoBuild).toBe(true);
    expect(config.missingCapabilities).toEqual([]);
  });

  it("detects pkg-config providers for openblas and fftw", () => {
    const execFileSyncImpl = vi.fn((command: string, args: string[]) => {
      expect(command).toBe("pkg-config");
      const [pkg, flag] = args;
      if (pkg === "openblas" && flag === "--libs") {
        return "-L/opt/openblas/lib -lopenblas";
      }
      if (pkg === "openblas" && flag === "--cflags") {
        return "-I/opt/openblas/include";
      }
      if (pkg === "fftw3" && flag === "--libs") {
        return "-L/opt/fftw/lib -lfftw3";
      }
      if (pkg === "fftw3" && flag === "--cflags") {
        return "-I/opt/fftw/include -DUSE_FFTW";
      }
      throw new Error("not found");
    });

    const config = resolveNativeAddonConfig({
      env: {},
      execFileSyncImpl,
    });

    expect(config.detectedPackages).toEqual({
      blas: "openblas",
      lapack: "openblas",
      fft: "fftw3",
    });
    expect(config.libraries).toEqual([
      "-L/opt/openblas/lib",
      "-lopenblas",
      "-L/opt/fftw/lib",
      "-lfftw3",
    ]);
    expect(config.linkerFlags).toEqual([
      "-Wl,-rpath,/opt/openblas/lib",
      "-Wl,-rpath,/opt/fftw/lib",
    ]);
    expect(config.includeDirs).toEqual([
      "/opt/openblas/include",
      "/opt/fftw/include",
    ]);
    expect(config.cflagsCc).toEqual(["-march=native", "-DUSE_FFTW"]);
    expect(config.defines).toEqual([
      "NUMBL_USE_OPENBLAS=1",
      "NUMBL_USE_FFTW=1",
      'NUMBL_BLAS_PROVIDER=\\"openblas\\"',
      'NUMBL_LAPACK_PROVIDER=\\"openblas\\"',
      'NUMBL_FFT_PROVIDER=\\"fftw3\\"',
    ]);
    expect(config.canAutoBuild).toBe(true);
  });

  it("accepts generic lapack plus fftw pkg-config output when the link line covers BLAS", () => {
    const execFileSyncImpl = vi.fn((command: string, args: string[]) => {
      expect(command).toBe("pkg-config");
      const [pkg, flag] = args;
      if (pkg === "lapack" && flag === "--libs") {
        return "-L/opt/lapack/lib -llapack -lblas";
      }
      if (pkg === "lapack" && flag === "--cflags") {
        return "-I/opt/lapack/include";
      }
      if (pkg === "fftw3" && flag === "--libs") {
        return "-L/opt/fftw/lib -lfftw3";
      }
      if (pkg === "fftw3" && flag === "--cflags") {
        return "-I/opt/fftw/include";
      }
      throw new Error("not found");
    });

    const config = resolveNativeAddonConfig({
      env: {},
      execFileSyncImpl,
    });

    expect(config.detectedPackages).toEqual({
      blas: null,
      lapack: "lapack",
      fft: "fftw3",
    });
    expect(config.libraries).toEqual([
      "-L/opt/lapack/lib",
      "-llapack",
      "-lblas",
      "-L/opt/fftw/lib",
      "-lfftw3",
    ]);
    expect(config.linkerFlags).toEqual([
      "-Wl,-rpath,/opt/lapack/lib",
      "-Wl,-rpath,/opt/fftw/lib",
    ]);
    expect(config.defines).toEqual([
      "NUMBL_USE_FFTW=1",
      'NUMBL_BLAS_PROVIDER=\\"blas\\"',
      'NUMBL_LAPACK_PROVIDER=\\"lapack\\"',
      'NUMBL_FFT_PROVIDER=\\"fftw3\\"',
    ]);
    expect(config.selectedProviders).toEqual({
      blas: "blas",
      lapack: "lapack",
      fft: "fftw3",
    });
    expect(config.resolvedCapabilities).toEqual({
      blas: true,
      lapack: true,
      fft: true,
      fftBackend: "fftw3",
      usesOpenBLAS: false,
    });
    expect(config.canAutoBuild).toBe(true);
    expect(config.missingCapabilities).toEqual([]);
  });

  it("detects blis plus libflame plus fftw pkg-config providers", () => {
    const execFileSyncImpl = vi.fn((command: string, args: string[]) => {
      expect(command).toBe("pkg-config");
      const [pkg, flag] = args;
      if (pkg === "blis" && flag === "--libs") {
        return "-L/opt/blis/lib -lblis";
      }
      if (pkg === "blis" && flag === "--cflags") {
        return "-I/opt/blis/include";
      }
      if (pkg === "libflame" && flag === "--libs") {
        return "-L/opt/flame/lib -lflame";
      }
      if (pkg === "libflame" && flag === "--cflags") {
        return "-I/opt/flame/include";
      }
      if (pkg === "fftw3" && flag === "--libs") {
        return "-L/opt/fftw/lib -lfftw3";
      }
      if (pkg === "fftw3" && flag === "--cflags") {
        return "-I/opt/fftw/include";
      }
      throw new Error("not found");
    });

    const config = resolveNativeAddonConfig({
      env: {},
      execFileSyncImpl,
    });

    expect(config.detectedPackages).toEqual({
      blas: "blis",
      lapack: "libflame",
      fft: "fftw3",
    });
    expect(config.libraries).toEqual([
      "-L/opt/blis/lib",
      "-lblis",
      "-L/opt/flame/lib",
      "-lflame",
      "-L/opt/fftw/lib",
      "-lfftw3",
    ]);
    expect(config.linkerFlags).toEqual([
      "-Wl,-rpath,/opt/blis/lib",
      "-Wl,-rpath,/opt/flame/lib",
      "-Wl,-rpath,/opt/fftw/lib",
    ]);
    expect(config.includeDirs).toEqual([
      "/opt/blis/include",
      "/opt/flame/include",
      "/opt/fftw/include",
    ]);
    expect(config.defines).toEqual([
      "NUMBL_USE_FFTW=1",
      'NUMBL_BLAS_PROVIDER=\\"blis\\"',
      'NUMBL_LAPACK_PROVIDER=\\"libflame\\"',
      'NUMBL_FFT_PROVIDER=\\"fftw3\\"',
    ]);
    expect(config.resolvedCapabilities).toEqual({
      blas: true,
      lapack: true,
      fft: true,
      fftBackend: "fftw3",
      usesOpenBLAS: false,
    });
    expect(config.canAutoBuild).toBe(true);
    expect(config.missingCapabilities).toEqual([]);
  });

  it("normalizes distro-specific pkg-config aliases for BLIS/libFLAME providers", () => {
    const execFileSyncImpl = vi.fn((command: string, args: string[]) => {
      expect(command).toBe("pkg-config");
      const [pkg, flag] = args;
      if (pkg === "blis" && flag === "--libs") {
        throw new Error("not found");
      }
      if (pkg === "blis-serial" && flag === "--libs") {
        return "-L/opt/blis/lib -lblis";
      }
      if (pkg === "blis-serial" && flag === "--cflags") {
        return "-I/opt/blis/include";
      }
      if (pkg === "libflame" && flag === "--libs") {
        return "-L/opt/flame/lib -lflame";
      }
      if (pkg === "libflame" && flag === "--cflags") {
        return "-I/opt/flame/include";
      }
      if (pkg === "fftw3" && flag === "--libs") {
        return "-L/opt/fftw/lib -lfftw3";
      }
      if (pkg === "fftw3" && flag === "--cflags") {
        return "-I/opt/fftw/include";
      }
      throw new Error("not found");
    });

    const config = resolveNativeAddonConfig({
      env: {},
      execFileSyncImpl,
    });

    expect(config.detectedPackages).toEqual({
      blas: "blis-serial",
      lapack: "libflame",
      fft: "fftw3",
    });
    expect(config.selectedProviders).toEqual({
      blas: "blis",
      lapack: "libflame",
      fft: "fftw3",
    });
    expect(config.defines).toEqual([
      "NUMBL_USE_FFTW=1",
      'NUMBL_BLAS_PROVIDER=\\"blis\\"',
      'NUMBL_LAPACK_PROVIDER=\\"libflame\\"',
      'NUMBL_FFT_PROVIDER=\\"fftw3\\"',
    ]);
    expect(config.canAutoBuild).toBe(true);
  });

  it("does not auto-build from incomplete explicit library overrides", () => {
    const config = resolveNativeAddonConfig({
      env: {
        NUMBL_NATIVE_LIBS: "-lopenblas",
      },
      execFileSyncImpl: vi.fn(),
    });

    expect(config.libraries).toEqual(["-lopenblas"]);
    expect(config.linkerFlags).toEqual([]);
    expect(config.resolvedCapabilities).toEqual({
      blas: true,
      lapack: true,
      fft: false,
      fftBackend: null,
      usesOpenBLAS: true,
    });
    expect(config.canAutoBuild).toBe(false);
    expect(config.missingCapabilities).toEqual(["FFT"]);
  });

  it("accepts explicit ducc0 fallback metadata for native builds", () => {
    const config = resolveNativeAddonConfig({
      env: {
        NUMBL_NATIVE_LIBS: "-L/opt/flame/lib -lflame -L/opt/blis/lib -lblis",
        NUMBL_NATIVE_DUCC0_INCLUDE_DIRS: "/opt/ducc/src",
        NUMBL_NATIVE_FFT_BACKEND: "ducc0",
        NUMBL_NATIVE_BLAS_PROVIDER: "blis",
        NUMBL_NATIVE_LAPACK_PROVIDER: "libflame",
        NUMBL_NATIVE_FFT_PROVIDER: "ducc0",
        NUMBL_NATIVE_PROVIDER_HINT: "local-blis-libflame-ducc0",
      },
      execFileSyncImpl: vi.fn(),
    });

    expect(config.includeDirs).toContain("/opt/ducc/src");
    expect(config.linkerFlags).toEqual([
      "-Wl,-rpath,/opt/flame/lib",
      "-Wl,-rpath,/opt/blis/lib",
    ]);
    expect(config.selectedProviders).toEqual({
      blas: "blis",
      lapack: "libflame",
      fft: "ducc0",
    });
    expect(config.defines).toEqual([
      "NUMBL_USE_DUCC0=1",
      'NUMBL_BLAS_PROVIDER=\\"blis\\"',
      'NUMBL_LAPACK_PROVIDER=\\"libflame\\"',
      'NUMBL_FFT_PROVIDER=\\"ducc0\\"',
    ]);
    expect(config.providerHint).toBe("local-blis-libflame-ducc0");
    expect(config.resolvedCapabilities).toEqual({
      blas: true,
      lapack: true,
      fft: true,
      fftBackend: "ducc0",
      usesOpenBLAS: false,
    });
    expect(config.canAutoBuild).toBe(true);
  });
});

describe("native addon install", () => {
  it("tries heuristic native builds when pkg-config metadata is unavailable", () => {
    const spawnSyncImpl = vi.fn(() => ({
      status: 0,
      error: undefined,
    }));
    const log = vi.fn();

    const result = runNativeAddonInstall({
      env: {},
      argv: [],
      execFileSyncImpl: vi.fn(() => {
        throw new Error("pkg-config missing");
      }),
      spawnSyncImpl,
      log,
      warn: vi.fn(),
    });

    expect(result.status).toBe("built");
    expect(spawnSyncImpl).toHaveBeenCalled();
    expect(spawnSyncImpl).toHaveBeenCalledWith(
      expect.any(String),
      expect.arrayContaining(["rebuild"]),
      expect.objectContaining({
        env: expect.objectContaining({
          NUMBL_NATIVE_LIBS: "-lopenblas -llapack -lfftw3",
          NUMBL_NATIVE_BLAS_PROVIDER: "openblas",
          NUMBL_NATIVE_LAPACK_PROVIDER: "lapack",
          NUMBL_NATIVE_FFT_PROVIDER: "fftw3",
        }),
      })
    );
    expect(log).toHaveBeenCalledWith(
      expect.stringContaining("Attempting native addon build with heuristic providers")
    );
  });

  it("uses the local node-gyp helper when available", () => {
    const spawnSyncImpl = vi.fn(() => ({
      status: 0,
      error: undefined,
    }));

    const result = runNativeAddonInstall({
      env: {},
      argv: [],
      execFileSyncImpl: vi.fn((command: string, args: string[]) => {
        const [pkg, flag] = args;
        if (command !== "pkg-config") throw new Error("unexpected command");
        if (pkg === "openblas" && flag === "--libs") return "-lopenblas";
        if (pkg === "openblas" && flag === "--cflags") return "";
        if (pkg === "fftw3" && flag === "--libs") return "-lfftw3";
        if (pkg === "fftw3" && flag === "--cflags") return "";
        throw new Error("not found");
      }),
      existsSyncImpl: (path: string) =>
        path.endsWith("node_modules/node-gyp/bin/node-gyp.js"),
      spawnSyncImpl,
      log: vi.fn(),
      warn: vi.fn(),
    });

    expect(result.status).toBe("built");
    expect(result.buildTool).toBe("local node-gyp");
    expect(spawnSyncImpl).toHaveBeenCalledWith(
      process.execPath,
      [expect.stringContaining("node_modules/node-gyp/bin/node-gyp.js"), "rebuild"],
      expect.objectContaining({
        stdio: "inherit",
      })
    );
  });

  it("builds when providers are detected", () => {
    const spawnSyncImpl = vi.fn(() => ({
      status: 0,
      error: undefined,
    }));

    const result = runNativeAddonInstall({
      env: {},
      argv: [],
      execFileSyncImpl: vi.fn((command: string, args: string[]) => {
        const [pkg, flag] = args;
        if (command !== "pkg-config") throw new Error("unexpected command");
        if (pkg === "openblas" && flag === "--libs") return "-lopenblas";
        if (pkg === "openblas" && flag === "--cflags") return "";
        if (pkg === "fftw3" && flag === "--libs") return "-lfftw3";
        if (pkg === "fftw3" && flag === "--cflags") return "";
        throw new Error("not found");
      }),
      spawnSyncImpl,
      existsSyncImpl: () => false,
      log: vi.fn(),
      warn: vi.fn(),
    });

    expect(result.status).toBe("built");
    expect(result.buildTool).toBe("npm exec node-gyp");
    expect(spawnSyncImpl).toHaveBeenCalledWith(
      "npm",
      ["exec", "--yes", "node-gyp", "rebuild"],
      expect.objectContaining({
        stdio: "inherit",
      })
    );
  });

  it("forces an explicit build even when install-time skip env vars are set", () => {
    const spawnSyncImpl = vi.fn(() => ({
      status: 0,
      error: undefined,
    }));

    const result = runNativeAddonInstall({
      env: {
        NUMBL_SKIP_NATIVE_INSTALL: "1",
      },
      argv: ["--force"],
      execFileSyncImpl: vi.fn((command: string, args: string[]) => {
        const [pkg, flag] = args;
        if (command !== "pkg-config") throw new Error("unexpected command");
        if (pkg === "openblas" && flag === "--libs") return "-lopenblas";
        if (pkg === "openblas" && flag === "--cflags") return "";
        if (pkg === "fftw3" && flag === "--libs") return "-lfftw3";
        if (pkg === "fftw3" && flag === "--cflags") return "";
        throw new Error("not found");
      }),
      spawnSyncImpl,
      log: vi.fn(),
      warn: vi.fn(),
    });

    expect(result.status).toBe("built");
    expect(spawnSyncImpl).toHaveBeenCalled();
  });

  it("builds the local fallback dependencies when --with-deps is requested", () => {
    const spawnSyncImpl = vi.fn(() => ({
      status: 0,
      error: undefined,
    }));
    const buildNativeDepsImpl = vi.fn(() => ({
      status: "built",
      env: {
        NUMBL_NATIVE_INCLUDE_DIRS: "/tmp/fallback/include",
        NUMBL_NATIVE_LIBS: "-L/tmp/fallback/lib -lflame -lblis",
        NUMBL_NATIVE_DUCC0_INCLUDE_DIRS: "/tmp/fallback/ducc/src",
        NUMBL_NATIVE_FFT_BACKEND: "ducc0",
        NUMBL_NATIVE_PROVIDER_HINT: "local-blis-libflame-ducc0",
        NUMBL_NATIVE_BLAS_PROVIDER: "blis",
        NUMBL_NATIVE_LAPACK_PROVIDER: "libflame",
        NUMBL_NATIVE_FFT_PROVIDER: "ducc0",
      },
    }));
    const log = vi.fn();

    const result = runNativeAddonInstall({
      env: {},
      argv: ["--with-deps"],
      execFileSyncImpl: vi.fn(() => {
        throw new Error("pkg-config missing");
      }),
      buildNativeDepsImpl,
      spawnSyncImpl,
      existsSyncImpl: () => false,
      log,
      warn: vi.fn(),
    });

    expect(result.status).toBe("built");
    expect(buildNativeDepsImpl).toHaveBeenCalledOnce();
    expect(result.config?.includeDirs).toContain("/tmp/fallback/include");
    expect(result.config?.selectedProviders).toEqual({
      blas: "blis",
      lapack: "libflame",
      fft: "ducc0",
    });
    expect(spawnSyncImpl).toHaveBeenCalledWith(
      "npm",
      ["exec", "--yes", "node-gyp", "rebuild"],
      expect.objectContaining({
        env: expect.objectContaining({
          NUMBL_NATIVE_FFT_BACKEND: "ducc0",
          NUMBL_NATIVE_PROVIDER_HINT: "local-blis-libflame-ducc0",
        }),
      })
    );
    expect(log).toHaveBeenCalledWith(
      "Building local BLIS/libFLAME/ducc0 fallback dependencies..."
    );
  });

  it("emits provider debug logs when native debug mode is enabled", () => {
    const spawnSyncImpl = vi.fn(() => ({
      status: 0,
      error: undefined,
    }));
    const log = vi.fn();

    runNativeAddonInstall({
      env: {
        NUMBL_DEBUG_NATIVE: "1",
      },
      argv: [],
      execFileSyncImpl: vi.fn((command: string, args: string[]) => {
        const [pkg, flag] = args;
        if (command !== "pkg-config") throw new Error("unexpected command");
        if (pkg === "openblas" && flag === "--libs") return "-lopenblas";
        if (pkg === "openblas" && flag === "--cflags") return "";
        if (pkg === "fftw3" && flag === "--libs") return "-lfftw3";
        if (pkg === "fftw3" && flag === "--cflags") return "";
        throw new Error("not found");
      }),
      spawnSyncImpl,
      log,
      warn: vi.fn(),
    });

    expect(log).toHaveBeenCalledWith(
      expect.stringContaining("Native provider search order:")
    );
    expect(log).toHaveBeenCalledWith(
      expect.stringContaining("Native provider summary:")
    );
  });

  it("honors explicit runtime library path overrides", () => {
    const config = resolveNativeAddonConfig({
      env: {
        NUMBL_NATIVE_LIBS: "-L/opt/math/lib -lopenblas -lfftw3",
        NUMBL_NATIVE_RPATH_DIRS: ["/opt/math/lib", "/opt/alt/lib"].join(
          delimiter
        ),
      },
      execFileSyncImpl: vi.fn(),
    });

    expect(config.linkerFlags).toEqual([
      "-Wl,-rpath,/opt/math/lib",
      "-Wl,-rpath,/opt/alt/lib",
    ]);
  });
});
