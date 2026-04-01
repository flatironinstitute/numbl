% heatmap - Heatmap of a correlation-like matrix
% Demonstrates a heatmap with labeled axes

data = [1.0 0.8 0.3 0.1;
        0.8 1.0 0.5 0.2;
        0.3 0.5 1.0 0.7;
        0.1 0.2 0.7 1.0];

figure;
heatmap([1 2 3 4], [1 2 3 4], data);
title('Correlation Matrix');
