% fnint (Curve Fitting Toolbox) / ppint (Octave): antiderivative of a pp
% form, with F(breaks(1)) = 0.

% Piecewise linear: f = s on [0,1], f = 1 - s on [1,2] (s local to piece)
pp = mkpp([0 1 2], [1 0; -1 1]);
F = fnint(pp);

assert(strcmp(F.form, 'pp'));
assert(F.pieces == 2);
assert(F.order == 3);

% F(0) = 0; F(1) = integral of s over [0,1] = 1/2
assert(ppval(F, 0) == 0);
assert(abs(ppval(F, 1) - 0.5) < 1e-12);

% Continuity across the break and total integral:
% F(2) = 1/2 + integral of (1-s) over [0,1] = 1/2 + 1/2 = 1
assert(abs(ppval(F, 2) - 1) < 1e-12);
assert(abs(ppval(F, 1.5) - 0.875) < 1e-12);

% ppint is an alias with the same result
F2 = ppint(pp);
assert(abs(ppval(F2, 1.5) - ppval(F, 1.5)) < 1e-15);
assert(abs(ppval(F2, 2) - 1) < 1e-12);

% Constant pp (order 1): integral of 2 over [0,3] = 6
ppc = mkpp([0 3], [2]);
Fc = fnint(ppc);
assert(abs(ppval(Fc, 3) - 6) < 1e-12);
assert(abs(ppval(Fc, 1.5) - 3) < 1e-12);

% Round-trip with interp1 pp form: integral of y=x from 1 to 3 is 4
ppl = interp1([1 2 3], [1 2 3], 'linear', 'pp');
Fl = fnint(ppl);
assert(abs(ppval(Fl, 3) - 4) < 1e-12);

disp('SUCCESS');
