# My numbl project

This is a [numbl](https://numbl.org) project deployed to **GitHub Pages**. The
whole thing runs in your browser — open a `.m` file in the panel on the left and
click **Run**. You can edit any file (changes are kept in memory for your
session) and re-run it.

## Try it

- [`main.m`](main.m) — a short demo that prints values and draws a plot.
- [`stats.m`](stats.m) — a helper function called by `main.m`.

```matlab
x = linspace(0, 2*pi, 200);
plot(x, sin(x), x, cos(x));
legend('sin', 'cos');
```

## How this is built

On every push to `main`, the workflow in
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) bundles these
files with the numbl browser IDE and publishes the result to GitHub Pages — no
server, no build step you have to run yourself.

Edit `numbl-project.json` to change the site title and which file opens first.
