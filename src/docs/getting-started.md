# Getting Started

numbl is a numerical computing environment compatible with MATLAB syntax that runs in your browser or on the command line.

## Browser

Visit the [numbl web app](/) to use the browser-based IDE and REPL. No installation required. Create projects, edit files, and run code directly in your browser.

## Command Line

Install numbl globally via npm:

```bash
npm install -g numbl
```

Then run `.m` files or start an interactive REPL:

```bash
# Run a script
numbl run script.m

# Evaluate inline code
numbl eval "disp(2 + 3)"

# Start interactive REPL
numbl
```

## Quick Example

```matlab
x = linspace(0, 4*pi, 200);
y = sin(x) .* exp(-x/10);
fprintf('Peak value: %.4f\n', max(y));
plot(x, y, 'LineWidth', 2);
title('Damped sine wave');
xlabel('x'); ylabel('y');
```

## CLI Options

```
Usage: numbl <command> [options]

Commands:
  run <file.m>       Run a .m file
  eval "<code>"      Evaluate inline code
  run-tests [dir]    Run .m test scripts (default: numbl_test_scripts/)
  build-addon        Build native LAPACK addon (--fast-math to enable -ffast-math)
  build-site <dir>   Build a static, browser-runnable site for a numbl project
  info               Print machine-readable info (JSON)
  list-builtins      List available built-in functions
  (no command)       Start interactive REPL

Global options:
  --version, -V      Print version and exit
  --help, -h         Print this help message

Options (for REPL):
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)

Options (for run and eval):
  --dump-js <file>   Write JS-JIT-generated JavaScript to file
  --dump-c <file>    Write C-JIT-generated C to file (--opt 2)
  --dump-ast         Print AST as JSON
  --verbose          Detailed logging to stderr
  --stream           NDJSON output mode
  --path <dir>       Add extra workspace directory
  --plot             Enable plot server
  --plot-port <port> Set plot server port (implies --plot)
  --opt <mode>       Optimization mode (default: 1)
                       0  interpreter (no JIT)
                       1  JS-JIT: type-specialize hot functions to JS
                       2  C-JIT: native kernels (Node only; falls back to JS-JIT)
  --fast-math        Build/load the native addon with -ffast-math: faster, but
                     FP results drift across --opt levels. Off by default.

Environment variables:
  NUMBL_PATH         Extra workspace directories (separated by :)
```

## Native Addon

For better performance with linear algebra and FFT, build the optional native addon:

```bash
numbl build-addon
```

This requires LAPACK and FFTW development libraries on your system. The addon provides native implementations for operations like matrix decompositions, eigenvalue problems, and FFT.

## Disclaimer

numbl is not affiliated with, endorsed by, or supported by MathWorks, Inc. MATLAB is a registered trademark of MathWorks, Inc.
