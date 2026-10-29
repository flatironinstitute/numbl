# Using numbl as a Library

numbl can be used as an npm library in both Node.js and browser applications.

## Install

```bash
npm install numbl
```

## Basic Usage

```js
import { executeCode } from "numbl";

const result = executeCode("disp(2 + 3)");
console.log(result.output); // ["5\n"]
```

## API

### `executeCode(source, options?, workspaceFiles?, mainFileName?, searchPaths?)`

Parses and executes `.m` source code. Returns an `ExecResult`:

- `output` -- array of printed lines
- `returnValue` -- the final `ans` value (`0` when no `ans` was set)
- `variableValues` -- all workspace variables after execution
- `plotInstructions` -- graphics commands (for rendering plots)

A sixth argument, `nativeBridge`, is still accepted for compatibility; it is deprecated in favor of the host's `nativeBridge` (see below).

### Options

Pass an `ExecOptions` object to control execution:

```js
const result = executeCode(code, {
  onOutput: text => console.log(text), // streaming output callback
  displayResults: true, // show expression results
  optimization: "1", // "0"=interpreter only, "1"=JS-JIT (default), "2"=C-JIT (Node only)
});
```

### Workspace Files

Define custom `.m` functions by passing them as workspace files:

```js
const result = executeCode("y = myfunc(3);", {}, [
  { name: "myfunc.m", source: "function r = myfunc(x)\nr = x^2;\nend" },
]);
console.log(result.variableValues.y); // 9
```

### Node Adapters (`numbl/node`)

In Node.js, real filesystem access, environment variables and directory scanning are provided by adapters. The implementations the numbl CLI uses are exported from the `numbl/node` entry point so hosts do not have to write their own:

```js
import { executeCode } from "numbl";
import { NodeFileIOAdapter, NodeSystemAdapter, scanMFiles } from "numbl/node";

// Run code that can see the .m files (including +packages and @classes)
// under some directory, as if it were on the MATLAB path:
const searchPaths = ["/path/to/matlab/code"];
const workspaceFiles = searchPaths.flatMap(p => scanMFiles(p));

const result = executeCode(
  "disp(myfunc(3))",
  {
    fileIO: new NodeFileIOAdapter(), // fopen/fileread/websave/unzip/...
    system: new NodeSystemAdapter(), // getenv/cd/computer/...
  },
  workspaceFiles,
  "eval.m",
  searchPaths
);
```

To back file I/O with some other store, implement the `FileIOAdapter` type exported from `numbl`. The core handles file identifiers itself (`fopen`, `fgetl`, `fread`, `fseek`, `fprintf` to a file, ...), so `readFile(path)`, `writeFile(path, bytes)`, `existsPath(path)` and `resolvePath(path)` are all it needs for those. Without random access, an open file is held in memory; providing `handles` (`openHandle`, `readAt`, `writeAt`, `size`, `close`, as `NodeFileIOAdapter` does) lets numbl read and write it in place; the other methods enable individual builtins such as `dir`, `copyfile` or `zip`. Files left open stay open across calls that pass the same adapter object. Adapters written against older releases, which implemented `fopen`/`fclose`/`freadBytes`/`fwriteBytes` themselves, still work; numbl now only uses those methods to read and write whole files.

### Hosts: platform capabilities

What the platform provides to a run (the numeric providers behind linear algebra, the qhull backends for `delaunay`/`convhull`, the C-JIT, native libraries for `.numbl.js` files, a WebGPU helper, and default file and system adapters) is collected in a host object (`NumblHost`). A run uses `options.host` if given, an `ExecSession` the host it was constructed with, and otherwise the process-default host (`defaultHost()`). `createHost()` makes a separate one, so two sessions in one process can have different capabilities:

```js
import { createHost, ExecSession } from "numbl";

const host = createHost(); // nothing but the TypeScript fallbacks
host.numeric.register({
  name: "mine",
  priority: 150,
  entries: { matmul: myMatmul }, // same signature as the built-in entry
});
const session = new ExecSession({ host });
```

A host's `ready` promise resolves once capabilities that load asynchronously are installed; await it before the first run when you fill a host yourself.

### Native Addon and C-JIT (`initNodePlatform`)

Optimization `"2"`, the native LAPACK addon and the qhull geometry backends need some Node-side setup, which the CLI does on startup. Library hosts call `initNodePlatform` once; it fills the process-default host, so later runs need nothing more:

```js
import { executeCode } from "numbl";
import { initNodePlatform } from "numbl/node";

await initNodePlatform();
executeCode(code, { optimization: "2" });
```

Pass `host: createHost()` to fill a separate host instead, and use the returned `host` with `executeCode(code, { host })` or `new ExecSession({ host })`. Other options: `native` (load the addon if built, default `true`), `fastMath` (default `false`), `cjit` (default `true`), `geometry` (load qhull, default `true`), `adapters` (make `NodeFileIOAdapter` and `NodeSystemAdapter` the host's defaults for runs that pass none, default `false`), and `mip` (install or refresh the mip core package from GitHub and return its directory in `searchPaths`, default `false`). Without koffi or a C compiler, `"2"` behaves like `"1"`. Code written for earlier releases, which passed the returned `nativeBridge` to `executeCode` or `ExecSession`, still works.

### Sessions (`ExecSession`)

`ExecSession` keeps variables, figures and search paths from one call to the next, as in a REPL or notebook:

```js
import { ExecSession } from "numbl";

const session = new ExecSession(); // or new ExecSession({ host })
session.execute("x = 3;");
session.execute("disp(x + 1)", { onOutput: t => process.stdout.write(t) });
```

### Error Handling

Errors throw a `RuntimeError` with file and line information:

```js
import { executeCode, RuntimeError } from "numbl";

try {
  executeCode("error('something went wrong')");
} catch (e) {
  if (e instanceof RuntimeError) {
    console.error(`${e.file}:${e.line}: ${e.message}`);
  }
}
```

## Examples

- [numbl-example-node](https://github.com/magland/numbl-example-node) -- Node.js usage
- [numbl-example-browser](https://github.com/magland/numbl-example-browser) -- Browser usage with Vite
- [numbl-image-filter](https://github.com/magland/numbl-image-filter) ([live demo](https://magland.github.io/numbl-image-filter/)) -- React app that filters images with a numbl script, passing image data in and out as tensors via a Web Worker
