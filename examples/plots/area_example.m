% area - Stacked area plot
% Demonstrates a stacked area chart with matrix data

x = 1:6;
Y = [3 4 2 5 3 4; 1 3 4 2 5 2; 2 1 3 3 2 3]';

figure;
area(x, Y);
title('Stacked Area Chart');
xlabel('Category');
ylabel('Value');
grid on;
