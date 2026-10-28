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

### `executeCode(source, options?, workspaceFiles?, mainFileName?)`

Parses and executes `.m` source code. Returns an `ExecResult`:

- `output` -- array of printed lines
- `returnValue` -- the final `ans` value
- `variableValues` -- all workspace variables after execution
- `plotInstructions` -- graphics commands (for rendering plots)

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

### Native Addon and C-JIT (`initNodePlatform`)

Optimization `"2"` and the native LAPACK addon need some Node-side setup, which the CLI does on startup. Library hosts call `initNodePlatform` once and pass the returned `nativeBridge` along:

```js
import { executeCode } from "numbl";
import { initNodePlatform } from "numbl/node";

const { nativeBridge } = await initNodePlatform();
executeCode(code, { optimization: "2" }, [], "script.m", [], nativeBridge);
```

Options: `native` (load the addon if built, default `true`), `fastMath` (default `false`), `cjit` (default `true`), and `mip` (install or refresh the mip core package from GitHub and return its directory in `searchPaths`, default `false`). Without koffi or a C compiler, `"2"` behaves like `"1"`.

### Sessions (`ExecSession`)

`ExecSession` keeps variables, figures and search paths from one call to the next, as in a REPL or notebook:

```js
import { ExecSession } from "numbl";

const session = new ExecSession({ nativeBridge }); // nativeBridge is optional
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
