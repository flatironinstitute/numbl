# Plotting

numbl supports a wide range of 2-D and 3-D plot types. In the browser, plots render directly in the IDE. On the command line, use `--plot` to open a plot server.

## Plot Types

| Category    | Functions                                                                         |
| ----------- | --------------------------------------------------------------------------------- |
| Line        | plot, plot3, fplot, fplot3                                                        |
| Scatter     | scatter, scatter3                                                                 |
| Bar         | bar, barh, bar3, bar3h                                                            |
| Surface     | surf, mesh                                                                        |
| Contour     | contour                                                                           |
| Image       | imagesc                                                                           |
| Histogram   | histogram, histogram2                                                             |
| Statistical | boxchart, swarmchart, swarmchart3                                                 |
| Other       | stairs, errorbar, area, semilogx, semilogy, loglog, piechart, donutchart, heatmap |

## Figure Management

figure, subplot, title, xlabel, ylabel, zlabel, sgtitle, legend, hold, grid, axis, view, colormap, colorbar, shading, close, clf, drawnow, pause

## Examples

### Line Plot

```matlab
x = linspace(0, 2*pi, 100);
plot(x, sin(x), 'b-', x, cos(x), 'r--');
title('Trig functions');
legend('sin', 'cos');
```

### Surface Plot

```matlab
[X, Y] = meshgrid(-2:0.1:2);
Z = X .* exp(-X.^2 - Y.^2);
surf(X, Y, Z);
colormap('jet');
colorbar;
```

### Subplots

```matlab
subplot(2, 1, 1);
plot(1:10, rand(1, 10));
title('Random data');

subplot(2, 1, 2);
bar(1:5, randi(10, 1, 5));
title('Bar chart');
```

### 3-D Scatter

```matlab
n = 200;
x = randn(n, 1);
y = randn(n, 1);
z = randn(n, 1);
scatter3(x, y, z, 20, z, 'filled');
colormap('cool');
```

## CLI Usage

To view plots when running from the command line:

```bash
numbl run --plot script.m
numbl --plot              # REPL with plot server
```

The plot server opens a browser window that updates as plot commands are executed.
