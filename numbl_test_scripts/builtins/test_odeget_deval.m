% Test odeget and deval

% ── odeget ──────────────────────────────────────────────────────────

opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-9);
assert(odeget(opts, 'RelTol') == 1e-6);
assert(odeget(opts, 'AbsTol') == 1e-9);

% odeget returns empty for missing fields
v = odeget(opts, 'MaxStep');
assert(isempty(v));

% ── deval ───────────────────────────────────────────────────────────

% Solve and get sol struct
sol = ode45(@(t,y) -y, [0 5], 1);

% Evaluate at specific points
xint = [0, 1, 2, 3, 4, 5];
yint = deval(sol, xint);
assert(size(yint, 1) == 1);
assert(size(yint, 2) == 6);

% Check accuracy at step boundary points (should be exact from interpolation)
for i = 1:length(xint)
    assert(abs(yint(i) - exp(-xint(i))) < 1e-4);
end

% Evaluate at a single point
y1 = deval(sol, 2.5);
assert(abs(y1 - exp(-2.5)) < 1e-4);

% System of equations
sol2 = ode45(@(t,y) [y(2); -y(1)], [0 2*pi], [1; 0]);
yint2 = deval(sol2, [0, pi/2, pi, 3*pi/2, 2*pi]);
assert(size(yint2, 1) == 2);
assert(size(yint2, 2) == 5);

% Check: y1 = cos(t), y2 = -sin(t) at t=pi/2
assert(abs(yint2(1,2) - cos(pi/2)) < 0.01);
assert(abs(yint2(2,2) - (-sin(pi/2))) < 0.01);

% deval with ode23 sol struct
sol3 = ode23(@(t,y) -y, [0 3], 1);
y3 = deval(sol3, 1.5);
assert(abs(y3 - exp(-1.5)) < 1e-3);

fprintf('=== deval Error Summary ===\n');
fprintf('y''=-y at grid:    max err: %.2e\n', max(abs(yint - exp(-xint))));
fprintf('y''=-y at 2.5:     err: %.2e\n', abs(y1 - exp(-2.5)));
fprintf('oscillator pi/2:  err: %.2e  %.2e\n', abs(yint2(1,2)-cos(pi/2)), abs(yint2(2,2)+sin(pi/2)));

disp('SUCCESS');
