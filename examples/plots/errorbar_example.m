% errorbar - Line plot with error bars
% Demonstrates vertical error bars

x = 1:10;
y = sin(x);
err = 0.1 + 0.1 * rand(1, 10);

figure;
errorbar(x, y, err);
title('Error Bar Plot');
xlabel('x');
ylabel('sin(x)');
grid on;
