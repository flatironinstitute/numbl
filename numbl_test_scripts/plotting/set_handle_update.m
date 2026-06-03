% Use set() to update line/marker handle data and a title handle's string,
% then read the values back through the handle to confirm they took effect.

% Line handle: update XData/YData via set (low-case property names).
f1 = line([0 1], [0 0], 'color', 'b', 'linewidth', 4);
set(f1, 'xdata', [0 10], 'ydata', [2 3]);
assert(isequal(f1.XData, [0 10]), 'XData should update');
assert(isequal(f1.YData, [2 3]), 'YData should update');

% Marker handle: scalar coordinates become single points.
f2 = line(0, 0, 'linestyle', 'none', 'marker', '.', 'markersize', 70);
set(f2, 'xdata', 5, 'ydata', 7);
assert(isequal(f2.XData, 5), 'scalar XData should update');
assert(isequal(f2.YData, 7), 'scalar YData should update');

% set should also update style properties.
set(f1, 'Color', 'r', 'LineWidth', 2);
assert(isequal(f1.Color, [1 0 0]), 'Color should resolve to red');
assert(f1.LineWidth == 2, 'LineWidth should update');

% Title handle: set the String and read it back.
tt = title('start');
set(tt, 'String', 'updated');
assert(strcmp(tt.String, 'updated'), 'title String should update');

% set on gca (a placeholder handle) is accepted as a no-op.
set(gca, 'FontSize', 14);

% Animation-style loop: repeatedly move a marker with set.
fb = line(0, 0, 'marker', '.');
for r = 1:5
    set(fb, 'xdata', r*2, 'ydata', r*r);
end
assert(isequal(fb.XData, 10), 'final XData after loop');
assert(isequal(fb.YData, 25), 'final YData after loop');

disp('SUCCESS');
