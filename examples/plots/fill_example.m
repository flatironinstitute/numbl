% fill - Filled polygons
% Regular polygons with an increasing number of sides, plus a reference circle.

colors = [0.00 0.45 0.74; 0.85 0.33 0.10; 0.93 0.69 0.13; 0.47 0.67 0.19];

figure;
hold on;
for k = 3:6
    th = linspace(0, 2*pi, k + 1);
    fill(3*k + cos(th), sin(th), colors(k - 2, :));
end
th = linspace(0, 2*pi, 100);
plot(21 + cos(th), sin(th), 'k-', 'LineWidth', 1.5);
hold off;
title('Polygons with 3 to 6 Sides');
axis equal;
