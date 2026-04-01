% boxchart - Box plot of grouped data
% Demonstrates box plots for multiple groups

groups = [ones(1,50), 2*ones(1,50), 3*ones(1,50)];
values = [randn(1,50) + 2, randn(1,50) * 1.5, randn(1,50) + 4];

figure;
boxchart(groups, values);
title('Box Chart by Group');
xlabel('Group');
ylabel('Value');
grid on;
