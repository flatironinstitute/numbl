% swarmchart - Swarm scatter plot
% Demonstrates jittered scatter plot showing data distribution

x = [ones(1,40), 2*ones(1,40), 3*ones(1,40)];
y = [randn(1,40), randn(1,40)*0.5 + 2, randn(1,40)*1.5 - 1];

figure;
swarmchart(x, y, 'filled');
title('Swarm Chart');
xlabel('Category');
ylabel('Value');
grid on;
