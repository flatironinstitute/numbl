% semilogx - Semilog plot (log x-axis)
% Demonstrates a plot with logarithmic x-axis

x = logspace(0, 4, 100);
y = sin(log10(x));

figure;
semilogx(x, y);
title('Semilog X Plot');
xlabel('x (log scale)');
ylabel('sin(log10(x))');
grid on;
