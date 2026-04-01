% scatter - Basic scatter plot
% Demonstrates a 2D scatter plot with random data

x = randn(1, 100);
y = 0.5 * x + 0.3 * randn(1, 100);

figure;
scatter(x, y, 'b');
title('Scatter Plot with Linear Trend');
xlabel('X');
ylabel('Y');
grid on;
