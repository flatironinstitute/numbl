# JavaScript user functions (`.numbl.js`)

A file named `name.numbl.js` on the path defines a function `name` that is implemented in JavaScript rather than MATLAB. It plays the role that a MEX file plays in MATLAB: numbl code calls it like any other function, and it can reach compiled code through WebAssembly or a native shared library. Like a `.m` file, it shadows a builtin of the same name, and a `.m` file of the same name shadows it.

## Defining a function

The file calls `register` once with the implementation:

```js
// myadd.numbl.js
register(function (args, nargout) {
  return args[0] + args[1];
});
```

`args` holds the call's arguments as numbl runtime values, and `nargout` is the number of outputs the caller requested. Scalars arrive as plain JS numbers; matrices are tensor objects with a column-major `data` array (a `Float64Array`) and a `shape`. The function returns:

- a single value when `nargout <= 1`;
- an array of values for multiple outputs;
- `undefined` if it has no outputs. Calling it with an output requested then raises "Too many output arguments."

A plain number may be returned for a scalar result; use `RTV.tensor(data, shape)`, `RTV.char(text)` and the other `RTV` constructors for anything else.

To provide text for `help name`, pass an object instead:

```js
// mystats.numbl.js
register({
  help:
    "MYSTATS  Mean and range of a vector.\n" +
    "   [m, r] = mystats(x) returns the mean m and the range r of x.",
  apply: function (args, nargout) {
    var x = args[0].data;
    // ... compute m and r ...
    if (nargout <= 1) return m;
    return [m, r];
  },
});
```

## Globals available to the file

| Name                           | Purpose                                                                                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `RTV`                          | Constructors for runtime values (`RTV.num`, `RTV.tensor`, `RTV.char`, `RTV.cell`, ...).                                                            |
| `RuntimeError`                 | Throw `new RuntimeError(msg)` to raise a numbl error.                                                                                              |
| `toNumber(v)`                  | Convert a scalar runtime value to a JS number.                                                                                                     |
| `callHandle(f, args, nargout)` | Call a numbl function handle passed in as an argument.                                                                                             |
| `importJS(name)`               | Load a library file `_name.numbl.js`, which `return`s its exports instead of calling `register`.                                                   |
| `wasm`                         | The WebAssembly instance named by a `// wasm: name` directive at the top of the file (`name.wasm` on the path), or `undefined`.                    |
| `native`                       | A native library loaded with koffi from a `// native: name` directive (`name.so`, `.dylib` or `.dll` next to the file), or `undefined`. Node only. |
| `gpu`                          | Synchronous access to WebGPU in the browser workers, or `null`.                                                                                    |

A function that uses `wasm` or `native` should keep a pure-JS fallback for when they are absent, for example in the browser (no native libraries) or when the binary has not been built.

## The older `resolve` form

Files written before the forms above call `register({ resolve })`, where `resolve(argTypes, nargout)` returns `{ outputTypes, apply }` (or `null` to reject the arguments). This form is still accepted, so existing files keep working. Converting one is mechanical: the body of `apply` becomes the registered function, and the `resolve` and `outputTypes` wrapper is dropped. A file whose `outputTypes` was `[]` (no outputs) should return `undefined`.

```js
// Before
register({
  resolve: function (argTypes, nargout) {
    var outs = [];
    for (var i = 0; i < Math.max(1, nargout); i++)
      outs.push({ kind: "unknown" });
    return {
      outputTypes: outs,
      apply: function (args, nargout) {
        return compute(args, nargout);
      },
    };
  },
});

// After
register(function (args, nargout) {
  return compute(args, nargout);
});
```

If `resolve` returned `null` for some arguments (for example, too few), the new form should check for that case and throw a `RuntimeError` with a message.
