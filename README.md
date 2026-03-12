# Numbl

Numbl is an open-source numerical computing environment that aims to be compatible with Matlab.

**Early stage project.** Numbl is under active development and new functionality is being added regularly.

## Try it in the browser

You can try numbl directly in the browser at <https://numbl.org> — no installation required. All execution happens locally in your browser. Note that the browser version has limited functionality and is slower than the desktop/command-line version.

## Embedding in web pages

Numbl scripts can be embedded in HTML and Markdown pages (including GitHub Pages). See the [numbl-embed-example](https://magland.github.io/numbl-embed-example/) for usage info and a [live demo](https://magland.github.io/numbl-embed-example/example1).

## Installation

```bash
npm install -g numbl
```

To enable fast linear algebra, build the native LAPACK addon:

```bash
# Prerequisites: C++ compiler, libopenblas-dev, libfftw3-dev (or equivalents for your OS)
numbl build-addon
```

## Usage

```bash
numbl                          # interactive REPL
numbl run script.m             # run a .m file
numbl eval "disp(eye(3))"      # evaluate inline code
numbl info                     # print info (JSON), including native addon status
numbl list-builtins            # list available built-in functions
numbl --help                   # show all commands and options
```

## Upgrading

```bash
npm install -g numbl@latest
```

Note: if you previously built the native addon, you'll need to run `numbl build-addon` again after upgrading.

## Authors

Jeremy Magland and Dan Fortunato, Center for Computational Mathematics, Flatiron Institute.

## License

Apache 2.0.

## Acknowledgements

See [acknowledgements.md](docs/acknowledgements.md).
