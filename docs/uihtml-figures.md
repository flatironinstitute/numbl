# Interactive HTML figures with `uihtml`

`uihtml` renders a self-contained HTML/CSS/JavaScript document inside a figure
and gives it a **two-way data bridge** to the interpreter. The same `.m` code
runs in **numbl** (browser IDE) and in **real MATLAB**, so you can build rich,
interactive figures (custom widgets, third-party JS visualizations, dashboards)
that work in both.

This is the tool to reach for when the axes/trace plotting model isn't enough —
when you want arbitrary HTML, your own interaction, or a JS charting library.

> **Where it works**
>
> - **numbl browser IDE** — full support: data in, events back to the interpreter.
> - **real MATLAB** — full support (this is a standard MATLAB component).
> - **numbl CLI (`numbl run --plot`)** — full support: the figure opens in a
>   browser window and events round-trip back into the interpreter, which stays
>   alive until you Ctrl+C. (The CLI **REPL** with `--plot` renders but does not
>   yet dispatch callbacks.)

---

## 1. Hello world

`uihtml` takes the HTML as a **string** (see the portability rules — do not pass
a file path). The idiomatic, portable form wraps it in a filling
`uigridlayout` so it fills the figure in MATLAB:

```matlab
fig = figure;
gl = uigridlayout(fig, [1 1], 'Padding', [0 0 0 0], ...
                  'RowHeight', {'1x'}, 'ColumnWidth', {'1x'});
uihtml(gl, 'HTMLSource', '<p style="font-family:sans-serif">Hello <b>world</b></p>');
```

For anything beyond trivial markup, keep the HTML in its own `.html` file and
read it as a string:

```matlab
here = fileparts(mfilename('fullpath'));
html = fileread(fullfile(here, 'myfigure.html'));
uihtml(gl, 'HTMLSource', html);
```

---

## 2. The `htmlComponent` contract (the JavaScript side)

Inside your HTML, define a global function `setup` that takes one argument, the
`htmlComponent` bridge object. The host calls `setup(htmlComponent)` once the
page has loaded:

```html
<script type="text/javascript">
  function setup(htmlComponent) {
    // htmlComponent.Data                      -> current data (get/set)
    // htmlComponent.addEventListener(n, fn)   -> listen for "DataChanged" and
    //                                            named events from the script
    // htmlComponent.sendEventToMATLAB(n, d)   -> send an event to the script
  }
</script>
```

`Data` is the **only** property that synchronizes. Anything sent across the
bridge is converted with `jsonencode` on the MATLAB/numbl side and
`JSON.parse` on the JS side (and back), so pass plain data: numbers, strings,
logicals, arrays, structs/objects, cell arrays.

---

## 3. Sending data into the figure (script → page)

Set `Data` and the page's `"DataChanged"` listener fires with the parsed value.

**Portable rule:** set `Data` at **construction** (name-value). To update it
later, assign `h.Data` and call `show(h)` to re-render (numbl renders at
construction and on `show`; it does not auto-re-render on a bare `h.Data = ...`).

```matlab
data = struct('title', 'Sales', 'values', [12 19 8 25]);
h = uihtml(gl, 'HTMLSource', html, 'Data', data);

% later:
h.Data = struct('title', 'Sales', 'values', [3 4 5 6]);
show(h);                 % re-render with the new data
```

```html
<script type="text/javascript">
  function setup(htmlComponent) {
    htmlComponent.addEventListener("DataChanged", function (event) {
      var d = htmlComponent.Data; // or event.Data
      render(d.title, d.values);
    });
  }
</script>
```

---

## 4. Reacting to the figure (page → script)

There are two channels back into the interpreter.

### 4a. Events: `sendEventToMATLAB` → `HTMLEventReceivedFcn`

The page sends a named event with a payload; your callback runs in the
interpreter. Register the callback at **construction** (name-value):

```matlab
uihtml(gl, 'HTMLSource', html, 'HTMLEventReceivedFcn', @onEvent);

function onEvent(src, event)
    switch event.HTMLEventName
        case 'Square'
            x = event.HTMLEventData;                  % payload from JS
            sendEventToHTMLSource(src, 'Result', x^2); % reply to the page
    end
end
```

```html
<script type="text/javascript">
  function setup(htmlComponent) {
    document.getElementById("go").addEventListener("click", function () {
      htmlComponent.sendEventToMATLAB("Square", 7);
    });
    // reply from MATLAB/numbl arrives as a named event:
    htmlComponent.addEventListener("Result", function (event) {
      document.getElementById("out").textContent = event.Data;
    });
  }
</script>
```

`sendEventToHTMLSource(src, name, data)` sends `name` back to the page, where it
fires `addEventListener(name, ...)` with `event.Data`. The `src` argument is the
component passed to your callback — pass it straight through.

### 4b. Data: JS sets `htmlComponent.Data` → `DataChangedFcn`

When JavaScript sets `htmlComponent.Data`, your `DataChangedFcn` runs (this does
**not** fire the JS `"DataChanged"` listener — that direction is script→page):

```matlab
uihtml(gl, 'HTMLSource', html, 'DataChangedFcn', @(src, event) disp(event.Data));
```

---

## 5. Callbacks: named, anonymous, and stateful

All three work in numbl and MATLAB:

```matlab
% named function
uihtml(gl, 'HTMLSource', html, 'HTMLEventReceivedFcn', @onEvent);

% anonymous function capturing a value
gain = 10;
uihtml(gl, 'HTMLSource', html, ...
       'HTMLEventReceivedFcn', @(src, ev) sendEventToHTMLSource(src, 'R', ev.HTMLEventData * gain));

% anonymous function forwarding to a helper, capturing an extra argument
offset = 100;
uihtml(gl, 'HTMLSource', html, ...
       'HTMLEventReceivedFcn', @(src, ev) onEvent(src, ev, offset));
```

**Stateful callbacks** that accumulate across clicks: capture a **handle-class
object** (anonymous functions capture by value, so a handle is how you share
mutable state across events):

```matlab
state = MyCounter();      % a `classdef MyCounter < handle` with a property
uihtml(gl, 'HTMLSource', html, ...
       'HTMLEventReceivedFcn', @(src, ev) bump(src, ev, state));

function bump(src, ev, state)
    state.n = state.n + ev.HTMLEventData;   % persists between callbacks
    sendEventToHTMLSource(src, 'Total', state.n);
end
```

---

## 6. Writing figures that work in BOTH MATLAB and numbl

Follow these rules and one file runs unchanged in both hosts:

1. **`HTMLSource` is a string, not a path.** numbl renders the markup string
   directly; it does not load file paths. Read your `.html` with `fileread`
   and pass the string. Make the document **fully self-contained** — inline all
   CSS/JS, no `<script src>` to a CDN, no external assets. (MATLAB's component
   sandbox also rejects CDNs, so this is required in both.)

2. **Set `Data` and callbacks at construction (name-value).** Use
   `uihtml(parent, 'HTMLSource', html, 'Data', d, 'HTMLEventReceivedFcn', @cb)`.
   numbl cannot intercept a post-construction property _set_ to re-render or
   re-register; to change data afterward use `h.Data = ...; show(h)`.

3. **Use a plain `figure` parent.** numbl's `figure` accepts only a numeric
   handle (`figure` or `figure(n)`) — `figure('Name', ...)` and `uifigure`
   are **not** supported. Plain `figure` works in both.

4. **Wrap in a filling `uigridlayout`** so the component fills the figure in
   MATLAB (default `Position` leaves big margins). numbl treats `uigridlayout`
   as a no-op (the pane already fills), so the same code is correct in both:

   ```matlab
   gl = uigridlayout(fig, [1 1], 'Padding', [0 0 0 0], ...
                     'RowHeight', {'1x'}, 'ColumnWidth', {'1x'});
   ```

5. **Pass plain data** (numbers, strings, logicals, arrays, structs, cells) so
   `jsonencode` handles it identically in both. For anything exotic, convert it
   yourself (e.g. `num2str`) before setting `Data`.

6. **Keep callbacks idempotent-friendly.** In numbl a new **Run** clears figures
   and disarms the previous run's callbacks (see §8). Don't rely on a figure
   from a prior run staying interactive across runs.

---

## 7. Complete round-trip example

`squarer.html`:

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
  </head>
  <body style="font-family: sans-serif">
    <input id="n" type="number" value="7" />
    <button id="go">Square in MATLAB</button>
    <div id="out">—</div>
    <script type="text/javascript">
      function setup(htmlComponent) {
        document.getElementById("go").addEventListener("click", function () {
          var x = Number(document.getElementById("n").value);
          htmlComponent.sendEventToMATLAB("Square", x);
        });
        htmlComponent.addEventListener("Result", function (event) {
          document.getElementById("out").textContent = event.Data;
        });
      }
    </script>
  </body>
</html>
```

`squarer.m` (runs in numbl IDE and MATLAB):

```matlab
function squarer
    here = fileparts(mfilename('fullpath'));
    html = fileread(fullfile(here, 'squarer.html'));

    fig = figure;
    gl = uigridlayout(fig, [1 1], 'Padding', [0 0 0 0], ...
                      'RowHeight', {'1x'}, 'ColumnWidth', {'1x'});
    uihtml(gl, 'HTMLSource', html, 'HTMLEventReceivedFcn', @onEvent);
end

function onEvent(src, event)
    if strcmp(event.HTMLEventName, 'Square')
        y = event.HTMLEventData ^ 2;
        fprintf('[MATLAB] %g^2 = %g\n', event.HTMLEventData, y);
        sendEventToHTMLSource(src, 'Result', y);
    end
end
```

---

## 8. numbl specifics and limitations

- **Reverse channel** fires in the numbl IDE, in MATLAB, and under
  `numbl run --plot` (the CLI keeps the runtime alive while the figure window is
  open). The CLI **REPL** with `--plot` renders but does not yet dispatch
  callbacks.
- **Re-render on data change** with `show(h)` — a bare `h.Data = ...` updates the
  property but numbl won't auto-redraw (MATLAB does).
- **Callback lifetime:** when a script registers a callback, numbl keeps the
  interpreter alive after the run so the figure can call back into it. Running
  another script (or the same one again) clears the figures and **disarms** the
  previous run's callbacks. The REPL disarms on the next command.
- **`HTMLSource` must be markup**, not a file path (MATLAB accepts both; numbl
  only the string — `fileread` keeps it portable).
- **No `uifigure` / `figure('Name',...)`** in numbl — use plain `figure`.

For how the bridge is implemented (the `uihtml` instruction, the injected
`htmlComponent` bootstrap, and the worker session that re-enters the
interpreter), see [developer_reference/uihtml.md](developer_reference/uihtml.md).
