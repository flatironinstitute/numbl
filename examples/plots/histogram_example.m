% histogram - Basic histogram
% Demonstrates a histogram of normally distributed data

x = randn(1, 1000);

figure;
histogram(x, 20);
title('Histogram of Normal Distribution');
xlabel('Value');
ylabel('Count');
grid on;
