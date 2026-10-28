# Deploying a Project to GitHub Pages

You can turn a folder of numbl files (`.m` scripts, `.md` docs, data files) into an interactive website where visitors browse the files, read rendered Markdown, and **run scripts entirely in their browser**, with no server required. The numbl browser IDE is bundled into the deploy alongside your files.

## Quick start

The fastest path is the starter template:

1. Create a new repository from the [numbl project template](https://github.com/flatironinstitute/numbl/tree/main/examples/numbl-project-template) (copy its files into a new repo, including `.github/workflows/deploy.yml`).
2. In your repo's **Settings → Pages**, set **Source** to **GitHub Actions**.
3. Add your `.m` and `.md` files and push to `main`.

Every push builds the site and publishes it to `https://<you>.github.io/<repo>/`.

## How it works

The deploy workflow calls the reusable action, which runs the `build-site` CLI command:

```yaml
- uses: actions/checkout@v4
- uses: flatironinstitute/numbl/.github/actions/build-site@main
  with:
    project-dir: .
- uses: actions/deploy-pages@v4
```

`build-site` copies the prebuilt browser IDE into the output directory and bundles your project files into a single `project.zip` that the IDE loads on startup. Visitors can edit and re-run files; their changes live in memory for the session (your committed files are always the source of truth).

## Building locally

You can produce the exact same output on your machine:

```bash
npx numbl build-site . --out _site
```

Then serve `_site/` with any static file server to preview it. (Running scripts in the browser needs cross-origin isolation for `SharedArrayBuffer`; a bundled service worker enables this automatically on GitHub Pages.)

For a project deployed under a repository subpath, pass the base path so assets resolve correctly (the GitHub Action does this for you):

```bash
npx numbl build-site . --out _site --base /my-repo/
```

## Configuration

Add a `numbl-project.json` at the project root to control the site:

```json
{
  "title": "My numbl project",
  "entry": "README.md"
}
```

- `title`: shown in the site header and the browser tab (defaults to the directory name).
- `entry`: the file opened first (defaults to `README.md`, then `main.m`, then the first script).
- `repository`: source repository URL, shown as a link in the site.
- `maxInitialOutputPanelHeight`: cap (px) on the initial height of the output panel so the figure panel below it starts larger (desktop layout). Omit for the default (half the window).
- `figures`: a list of `{ "id", "name", "entry" }` figure views. Each is served at `#figure/<id>` and shows the figure produced by running the `entry` script.

The `build-site` flags `--title`, `--entry`, `--repo-url` and `--max-initial-output-panel-height`, and the matching action inputs (`title:`, `entry:`, `repo-url:`, `max-initial-output-panel-height:`), override `numbl-project.json`. The action fills in `repo-url` and the base path from the repository when they are not given.

To exclude files from the bundle, add a `.numblignore` (gitignore-style globs):

```
*.bin
scratch/
**/*.tmp
```

`.git`, `.github`, and `node_modules` are always skipped.

## Notes and limitations

- Scripts run with the in-browser interpreter and JS JIT (`--opt 0`/`--opt 1`). The C JIT (`--opt 2`) is Node-only and is not available in the browser.
- Helper functions must be reachable on the path. The simplest approach is to keep functions called from your scripts in the project root.
- Large binary files inflate the bundle (and the initial download). Use `.numblignore` to omit anything visitors don't need.
