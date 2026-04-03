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
  optimization: 1, // 0=interpreter only, 1=JIT (default)
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
