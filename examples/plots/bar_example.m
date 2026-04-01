% bar - Basic bar chart
% Demonstrates a simple bar chart of categorical data

categories = [15 22 18 25 30 12];

figure;
bar(categories);
title('Monthly Widget Sales');
xlabel('Month');
ylabel('Units Sold');
grid on;
