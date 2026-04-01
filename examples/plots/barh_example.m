% barh - Horizontal bar chart
% Demonstrates a simple horizontal bar chart

categories = [15 22 18 25 30 12];

figure;
barh(categories);
title('Monthly Widget Sales');
xlabel('Units Sold');
ylabel('Month');
grid on;
