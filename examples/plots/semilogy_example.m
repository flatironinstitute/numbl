% semilogy - Semilog plot (log y-axis)
% Demonstrates a plot with logarithmic y-axis

x = 0:0.5:10;
y = exp(x);

figure;
semilogy(x, y);
title('Semilog Y Plot');
xlabel('x');
ylabel('exp(x) (log scale)');
grid on;
