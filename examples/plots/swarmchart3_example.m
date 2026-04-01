% swarmchart3 - 3D swarm scatter plot
% Demonstrates jittered 3D scatter plot

x = [ones(1,30), 2*ones(1,30), 3*ones(1,30)];
y = [ones(1,30), 2*ones(1,30), ones(1,30)];
z = [randn(1,30), randn(1,30)*0.5 + 2, randn(1,30)*1.5];

figure;
swarmchart3(x, y, z, 'filled');
title('3D Swarm Chart');
xlabel('X');
ylabel('Y');
zlabel('Z');
