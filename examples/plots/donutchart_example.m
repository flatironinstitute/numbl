% donutchart - Donut chart with labeled slices
% Demonstrates a donut chart (pie chart with hole)

data = [40 30 20 10];

figure;
donutchart(data, 'Python', 'JavaScript', 'TypeScript', 'Other');
title('Language Usage');
