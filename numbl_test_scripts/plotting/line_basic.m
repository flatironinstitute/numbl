% Smoke test for the line primitive: 2-D, 3-D, matrix, low-level, and
% property-after-creation forms.

% line(x, y) — single 2-D line
x = linspace(0, 10);
y = sin(x);
line(x, y);

% Matrix input draws one line per column
x2 = linspace(0, 10)';
y2 = [sin(x2) cos(x2)];
line(x2, y2);

% line(x, y, z) — 3-D line
t = linspace(0, 10*pi, 200);
line(sin(t), cos(t), t);

% No arguments — line from (0,0) to (1,1)
line;

% Name-Value styling
line([1 9], [2 12], 'Color', 'red', 'LineStyle', '--', 'LineWidth', 3);

% Low-level form draws a black line
line('XData', x, 'YData', y);

% Return handle and change properties after creation
pl = line([3 2], [15 12]);
pl.Color = 'green';
pl.LineStyle = '--';

disp('SUCCESS');
