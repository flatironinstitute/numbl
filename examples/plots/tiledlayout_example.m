% tiledlayout - tiled chart layout for multiple plots
% Four views of a damped oscillation arranged in a 2-by-2 tiled layout.

t = linspace(0, 10, 200);
y = exp(-0.3 * t) .* cos(2 * pi * t);

figure;
tiledlayout(2, 2);

nexttile;
plot(t, y);
title('Signal');
xlabel('t');

nexttile;
plot(t, exp(-0.3 * t), 'r--');
hold on;
plot(t, -exp(-0.3 * t), 'r--');
plot(t, y, 'b');
title('With Envelope');
xlabel('t');

nexttile;
plot(y(1:end-1), y(2:end));
title('Phase Portrait');
xlabel('y(n)');
ylabel('y(n+1)');
axis equal;

nexttile;
scatter(t(1:5:end), y(1:5:end));
title('Sampled');
xlabel('t');
