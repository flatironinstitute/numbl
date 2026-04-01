% Test dop853 - Explicit Runge-Kutta method of order 8
% Note: dop853 is not a MATLAB builtin. Run this test only in numbl.

% Simple scalar ODE: y' = 2t, y(0) = 0 => y = t^2
[t, y] = dop853(@(t,y) 2*t, [0 5], 0);
assert(length(t) > 2);
assert(abs(y(end) - 25) < 1e-10);

% Exponential decay: y' = -y, y(0) = 1 => y = exp(-t)
[t, y] = dop853(@(t,y) -y, [0 5], 1);
assert(abs(y(end) - exp(-5)) < 1e-6);

% Harmonic oscillator: y1 = cos(t), y2 = -sin(t)
[t, y] = dop853(@(t,y) [y(2); -y(1)], [0 2*pi], [1; 0]);
assert(abs(y(end, 1) - 1) < 1e-4);
assert(abs(y(end, 2)) < 1e-4);

% Specified output times (dense output)
tspan = linspace(0, 1, 50);
[t, y] = dop853(@(t,y) -y, tspan, 1);
assert(length(t) == 50);
max_interp_err = max(abs(y' - exp(-t')));
assert(max_interp_err < 1e-8);

% Tight tolerances -- 8th order should be very accurate
opts = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
[t, y] = dop853(@(t,y) -y, [0 5], 1, opts);
assert(abs(y(end) - exp(-5)) < 1e-12);

% Sol struct output
sol = dop853(@(t,y) -y, [0 5], 1);
assert(strcmp(sol.solver, 'dop853'));
assert(abs(sol.x(end) - 5) < 1e-15);

% deval with dop853 sol struct
yint = deval(sol, [1, 2, 3]);
for i = 1:3
    assert(abs(yint(i) - exp(-i)) < 1e-4);
end

% Output sizes
[t, y] = dop853(@(t,y) [y(2); -y(1)], [0 1], [1; 0]);
assert(size(t, 2) == 1);
assert(size(y, 2) == 2);

% Print error summary
fprintf('=== dop853 Error Summary ===\n');

[~, y1] = dop853(@(t,y) 2*t, [0 5], 0);
fprintf('y''=2t       endpoint err: %.2e\n', abs(y1(end) - 25));

[~, y2] = dop853(@(t,y) -y, [0 5], 1);
fprintf('y''=-y       endpoint err: %.2e\n', abs(y2(end) - exp(-5)));

[~, y3] = dop853(@(t,y) [y(2); -y(1)], [0 2*pi], [1; 0]);
fprintf('oscillator  endpoint err: %.2e  %.2e\n', abs(y3(end,1)-1), abs(y3(end,2)));

fprintf('interp max err:          %.2e\n', max_interp_err);

opts2 = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
[~, y4] = dop853(@(t,y) -y, [0 5], 1, opts2);
fprintf('tight tol   endpoint err: %.2e\n', abs(y4(end) - exp(-5)));

disp('SUCCESS');
