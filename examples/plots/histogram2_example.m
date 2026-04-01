% histogram2 - Bivariate histogram
% Demonstrates a 2D histogram of correlated random data

x = randn(1, 5000);
y = 0.6 * x + 0.8 * randn(1, 5000);

figure;
histogram2(x, y, 15);
title('Bivariate Histogram');
xlabel('X');
ylabel('Y');
zlabel('Count');
