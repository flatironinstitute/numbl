% Test ode23 - Bogacki-Shampine ODE solver

% Simple scalar ODE: y' = 2t, y(0) = 0 => y = t^2
[t, y] = ode23(@(t,y) 2*t, [0 5], 0);
assert(length(t) > 2);
assert(abs(y(end) - 25) < 1e-4);

% Exponential decay: y' = -y, y(0) = 1 => y = exp(-t)
[t, y] = ode23(@(t,y) -y, [0 5], 1);
assert(abs(y(end) - exp(-5)) < 1e-3);

% System: harmonic oscillator
[t, y] = ode23(@(t,y) [y(2); -y(1)], [0 2*pi], [1; 0]);
assert(abs(y(end, 1) - 1) < 0.02);
assert(abs(y(end, 2)) < 0.02);

% Specified output times
tspan = linspace(0, 1, 20);
[t, y] = ode23(@(t,y) -y, tspan, 1);
assert(length(t) == 20);
for i = 1:length(t)
    assert(abs(y(i) - exp(-t(i))) < 0.02);
end

% Tight tolerances
opts = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[t, y] = ode23(@(t,y) -y, [0 5], 1, opts);
assert(abs(y(end) - exp(-5)) < 1e-7);

% Sol struct output
sol = ode23(@(t,y) -y, [0 5], 1);
assert(strcmp(sol.solver, 'ode23'));
assert(abs(sol.x(end) - 5) < 1e-15);

% Output sizes
[t, y] = ode23(@(t,y) [y(2); -y(1)], [0 1], [1; 0]);
assert(size(t, 2) == 1);
assert(size(y, 2) == 2);

% Print error summary
fprintf('=== ode23 Error Summary ===\n');
[~, y1] = ode23(@(t,y) -y, [0 5], 1);
fprintf('y''=-y       endpoint err: %.2e\n', abs(y1(end) - exp(-5)));

[~, y2] = ode23(@(t,y) [y(2); -y(1)], [0 2*pi], [1; 0]);
fprintf('oscillator  endpoint err: %.2e  %.2e\n', abs(y2(end,1)-1), abs(y2(end,2)));

opts2 = odeset('RelTol', 1e-8, 'AbsTol', 1e-10);
[~, y3] = ode23(@(t,y) -y, [0 5], 1, opts2);
fprintf('tight tol   endpoint err: %.2e\n', abs(y3(end) - exp(-5)));

disp('SUCCESS');
