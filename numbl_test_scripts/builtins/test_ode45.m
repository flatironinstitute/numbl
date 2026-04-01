% Test ode45 - Dormand-Prince ODE solver

% Simple scalar ODE: y' = 2t, y(0) = 0 => y = t^2
[t, y] = ode45(@(t,y) 2*t, [0 5], 0);
assert(length(t) > 2);
assert(abs(y(end) - 25) < 1e-10);
assert(abs(t(1)) < 1e-15);
assert(abs(t(end) - 5) < 1e-15);

% Exponential decay: y' = -y, y(0) = 1 => y = exp(-t)
[t, y] = ode45(@(t,y) -y, [0 5], 1);
assert(abs(y(end) - exp(-5)) < 1e-5);

% System of equations: y1' = y2, y2' = -y1 (harmonic oscillator)
% y1(0) = 1, y2(0) = 0 => y1 = cos(t), y2 = -sin(t)
[t, y] = ode45(@(t,y) [y(2); -y(1)], [0 2*pi], [1; 0]);
assert(abs(y(end, 1) - 1) < 1e-3);
assert(abs(y(end, 2)) < 1e-3);

% Test with specified output times
tspan = linspace(0, 1, 20);
[t, y] = ode45(@(t,y) -y, tspan, 1);
assert(length(t) == 20);
for i = 1:length(t)
    assert(abs(y(i) - exp(-t(i))) < 1e-7);
end

% Test with tolerances via odeset
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t, y] = ode45(@(t,y) -y, [0 5], 1, opts);
assert(abs(y(end) - exp(-5)) < 1e-9);

% Test output sizes: t should be column, y should be nPoints x neq
[t, y] = ode45(@(t,y) [y(2); -y(1)], [0 1], [1; 0]);
assert(size(t, 2) == 1);  % column vector
assert(size(y, 2) == 2);  % two columns for two equations
assert(size(y, 1) == size(t, 1));

% Test sol struct output
sol = ode45(@(t,y) -y, [0 5], 1);
assert(strcmp(sol.solver, 'ode45'));
assert(abs(sol.x(1)) < 1e-15);
assert(abs(sol.x(end) - 5) < 1e-15);
assert(size(sol.y, 1) == 1);  % 1 equation
assert(size(sol.y, 2) == length(sol.x));

% Van der Pol with mu=1 (nonstiff)
vdp = @(t,y) [y(2); (1 - y(1)^2)*y(2) - y(1)];
[t, y] = ode45(vdp, [0 20], [2; 0]);
assert(length(t) > 10);
assert(all(abs(y(:,1)) < 3));  % solution stays bounded

% Test event detection: bouncing ball y'' = -g, detect y = 0
opts_ev = odeset('Events', @ball_event);
[t, y, te, ye, ie] = ode45(@(t,y) [y(2); -9.81], [0 10], [10; 0], opts_ev);
% Ball should hit ground around t = sqrt(2*10/9.81) ~ 1.428
assert(length(te) >= 1);
assert(abs(te(1) - sqrt(2*10/9.81)) < 1e-6);
assert(abs(ye(1,1)) < 1e-6);  % y ≈ 0 at impact

% Print error summary for comparison with MATLAB
fprintf('=== Error Summary ===\n');

[t1, y1] = ode45(@(t,y) 2*t, [0 5], 0);
fprintf('y''=2t       endpoint err: %.2e\n', abs(y1(end) - 25));

[t2, y2] = ode45(@(t,y) -y, [0 5], 1);
fprintf('y''=-y       endpoint err: %.2e\n', abs(y2(end) - exp(-5)));

[t3, y3] = ode45(@(t,y) [y(2); -y(1)], [0 2*pi], [1; 0]);
fprintf('oscillator  endpoint err: %.2e  %.2e\n', abs(y3(end,1)-1), abs(y3(end,2)));

tspan4 = linspace(0, 1, 20);
[t4, y4] = ode45(@(t,y) -y, tspan4, 1);
fprintf('interp max err:          %.2e\n', max(abs(y4 - exp(-t4))));

opts5 = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t5, y5] = ode45(@(t,y) -y, [0 5], 1, opts5);
fprintf('tight tol   endpoint err: %.2e\n', abs(y5(end) - exp(-5)));

fprintf('event time  err:         %.2e\n', abs(te(1) - sqrt(2*10/9.81)));

disp('SUCCESS');

function [value, isterminal, direction] = ball_event(t, y)
    value = y(1);       % detect y(1) = 0
    isterminal = 1;     % stop integration
    direction = -1;     % only decreasing
end
