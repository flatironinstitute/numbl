% loglog - Log-log scale plot
% Demonstrates a plot with logarithmic axes on both x and y

x = logspace(0, 4, 100);
y = x .^ 2;

figure;
loglog(x, y);
title('Log-Log Plot');
xlabel('x');
ylabel('x^2');
grid on;
