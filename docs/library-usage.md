# Using numbl as a library

numbl can be used as an npm library in both Node.js and browser applications.

## Install

```bash
npm install numbl
```

## Basic usage

```js
import { executeCode } from "numbl";

const result = executeCode("disp(2 + 3)");
console.log(result.output); // ["5\n"]
```

## API

### `executeCode(source, options?, workspaceFiles?, mainFileName?)`

Parses and executes `.m` source code. Returns an `ExecResult`:

- `output` — array of printed lines
- `returnValue` — the final `ans` value
- `variableValues` — all workspace variables after execution
- `plotInstructions` — graphics commands (for rendering plots)

### Options

Pass an `ExecOptions` object to control execution:

```js
const result = executeCode(code, {
  onOutput: text => console.log(text), // streaming output callback
  displayResults: true, // show expression results
  optimization: "1", // "0"=interpreter only, "1"=JS-JIT (default), "2"=C-JIT (Node only)
});
```

### Workspace files

Define custom `.m` functions by passing them as workspace files:

```js
const result = executeCode("y = myfunc(3);", {}, [
  { name: "myfunc.m", source: "function r = myfunc(x)\nr = x^2;\nend" },
]);
console.log(result.variableValues.y); // 9
```

### Node adapters (`numbl/node`)

In Node.js, real filesystem access, env vars, and directory scanning are
provided by adapters. The implementations the numbl CLI uses are exported
from the `numbl/node` entry point so hosts don't have to write their own:

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

### Error handling

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

- [numbl-example-node](https://github.com/magland/numbl-example-node) — Node.js usage
- [numbl-example-browser](https://github.com/magland/numbl-example-browser) — Browser usage with Vite
- [numbl-image-filter](https://github.com/magland/numbl-image-filter) ([live demo](https://magland.github.io/numbl-image-filter/)) — React app that filters images with a numbl script, passing image data in and out as tensors via a Web Worker
