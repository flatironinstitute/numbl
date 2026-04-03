# Numbl

A MATLAB-compatible numerical computing environment with 400+ built-in functions. Runs in your browser or on the command line.

[![numbl REPL](docs/repl-preview.svg)](https://numbl.org/embed-repl)

**[Documentation](https://numbl.org/docs)** | **[Browser IDE](https://numbl.org)** | **[REPL](https://numbl.org/embed-repl)** | **[Plot Gallery](https://numbl.org/gallery)**

## Quick Start

Try it in the browser at [numbl.org](https://numbl.org) -- no installation required.

Or use the CLI:

```bash
npx numbl                      # interactive REPL
npx numbl eval "disp(eye(3))"  # evaluate inline code
npx numbl run script.m         # run a .m file
```

Install globally for regular use:

```bash
npm install -g numbl
```

## Native Addon

For faster linear algebra, FFT, and C++ operations, build the optional native addon:

```bash
# Prerequisites: C++ compiler, libopenblas-dev, libfftw3-dev (or equivalents for your OS)
numbl build-addon
```

Rebuild after upgrading numbl (`npm install -g numbl@latest`).

## Documentation

Full documentation is available at **[numbl.org/docs](https://numbl.org/docs)**, covering:

- [Getting Started](https://numbl.org/docs/getting-started) -- installation, CLI options, native addon
- [Language Features](https://numbl.org/docs/language) -- operators, data types, control flow, classes
- [Built-in Functions](https://numbl.org/docs/builtins) -- 400+ functions by category
- [Plotting](https://numbl.org/docs/plotting) -- 2-D/3-D plots, examples, CLI plot server
- [Library Usage](https://numbl.org/docs/library) -- using numbl as an npm package
- [Differences from MATLAB](https://numbl.org/docs/differences) -- behavioral differences and limitations

## VS Code Extension

The [Numbl extension for VS Code](https://marketplace.visualstudio.com/items?itemName=jmagland.numbl) provides inline error diagnostics and a built-in figure viewer.

## Embedding

Numbl scripts can be embedded in HTML and Markdown pages. See the [numbl-embed-example](https://magland.github.io/numbl-embed-example/) for usage and a [live demo](https://magland.github.io/numbl-embed-example/example1).

## Authors

Jeremy Magland and Dan Fortunato, Center for Computational Mathematics, Flatiron Institute.

## License

Apache 2.0
