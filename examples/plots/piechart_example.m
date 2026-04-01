% piechart - Pie chart with labeled slices
% Demonstrates a pie chart with named slices

data = [35 25 20 15 5];

figure;
piechart(data, 'Rent', 'Food', 'Transport', 'Savings', 'Other');
title('Monthly Budget');
