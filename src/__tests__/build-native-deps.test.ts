import { afterEach, describe, expect, it } from "vitest";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

// @ts-expect-error Vitest resolves the script module at runtime.
import { buildNativeDeps } from "../../scripts/build-native-deps.mjs";

type SpawnCall = {
  command: string;
  args: string[];
  cwd: string | undefined;
  env: Record<string, string | undefined>;
};

describe("buildNativeDeps", () => {
  let tempRoot: string | null = null;

  afterEach(() => {
    if (tempRoot) {
      rmSync(tempRoot, { recursive: true, force: true });
      tempRoot = null;
    }
  });

  it("pins stable upstream refs for the local BLIS/libFLAME/ducc0 fallback", () => {
    tempRoot = mkdtempSync(join(tmpdir(), "numbl-native-deps-"));
    const prefix = join(tempRoot, "install");
    const calls: SpawnCall[] = [];

    const spawnSyncImpl = (
      command: string,
      args: string[],
      options: { cwd?: string; env?: Record<string, string | undefined> }
    ) => {
      calls.push({
        command,
        args: [...args],
        cwd: options.cwd,
        env: options.env ?? {},
      });

      if (command === "git" && args[0] === "clone") {
        const targetDir = args[args.length - 1];
        mkdirSync(join(targetDir, ".git"), { recursive: true });
      }

      if (command === "make" && args[0] === "install") {
        mkdirSync(join(prefix, "lib"), { recursive: true });
        if (String(options.cwd).includes(".numbl-native-build")) {
          writeFileSync(join(prefix, "lib", "libflame.a"), "");
        } else {
          writeFileSync(join(prefix, "lib", "libblis.a"), "");
        }
      }

      return { status: 0 };
    };

    const result = buildNativeDeps({
      env: {
        NUMBL_NATIVE_DEPS_ROOT: tempRoot,
        NUMBL_NATIVE_DEPS_PREFIX: prefix,
        NUMBL_NATIVE_DEPS_JOBS: "1",
        NUMBL_BLIS_UPSTREAM_ROOT: join(tempRoot, "custom-src", "blis"),
        NUMBL_LIBFLAME_UPSTREAM_ROOT: join(tempRoot, "custom-src", "libflame"),
        NUMBL_DUCC0_UPSTREAM_ROOT: join(tempRoot, "custom-src", "ducc"),
      },
      spawnSyncImpl,
    });

    const cloneCalls = calls.filter(
      call => call.command === "git" && call.args[0] === "clone"
    );

    expect(
      cloneCalls.some(
        call =>
          call.args.includes("https://github.com/flame/blis.git") &&
          call.args.includes("--branch") &&
          call.args.includes("2.0")
      )
    ).toBe(true);
    expect(
      cloneCalls.some(
        call =>
          call.args.includes("https://github.com/flame/libflame.git") &&
          call.args.includes("--branch") &&
          call.args.includes("5.2.0")
      )
    ).toBe(true);
    expect(
      cloneCalls.some(
        call =>
          call.args.includes("https://github.com/DiamonDinoia/ducc.git") &&
          call.args.includes("--branch") &&
          call.args.includes("ducc0")
      )
    ).toBe(true);

    expect(result.sourceRoots.blis).toBe(join(tempRoot, "custom-src", "blis"));
    expect(result.sourceRoots.libflame).toBe(
      join(tempRoot, "custom-src", "libflame")
    );
    expect(result.sourceRoots.ducc0).toBe(join(tempRoot, "custom-src", "ducc"));
    expect(result.env.NUMBL_NATIVE_PROVIDER_HINT).toBe(
      "local-blis-libflame-ducc0"
    );
    expect(result.env.NUMBL_NATIVE_BLAS_PROVIDER).toBe("blis");
    expect(result.env.NUMBL_NATIVE_LAPACK_PROVIDER).toBe("libflame");
    expect(result.env.NUMBL_NATIVE_FFT_PROVIDER).toBe("ducc0");

    const configuredBuildCall = calls.find(
      call =>
        call.command.endsWith("configure") &&
        call.env.CFLAGS?.includes("-fno-semantic-interposition")
    );
    expect(configuredBuildCall?.env.CFLAGS).toContain("-fno-fast-math");
    expect(configuredBuildCall?.env.CFLAGS).toContain("-fno-math-errno");
    expect(configuredBuildCall?.env.CFLAGS).toContain("-ffp-contract=on");
    expect(configuredBuildCall?.env.CFLAGS).toContain(
      "-fno-semantic-interposition"
    );
  });
});
