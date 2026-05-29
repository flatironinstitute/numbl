% Sign propagation through `/` so `sqrt(positive/positive)` stays real
% (number, not complex_or_number). Without this, downstream `>` etc.
% bail because comparison rejects complex_or_number scalars.
%
% Pattern is hot in chunkerfunc/chunkie's resolve loop:
%   err1 = sqrt(errs/errs0/k);   % all RHS positive
%   resol_speed_test = err1 > eps;
% which previously fell back to the interpreter for the whole loop.

eps_tol = 1e-6;
n = 20;
k = 16;
acc = 0;
for i = 1:n
    err1 = sqrt(0.5 / 0.5 / k);
    resol_speed_test = err1 > eps_tol;
    if resol_speed_test
        acc = acc + 1;
    end
end
assert(acc == n, 'every iteration should pass the threshold');

% Also exercise ElemDiv (the `./` operator), which shares the rule.
% Real-tensor base, real scalar divisor, all-positive — the result
% must stay tensor-real, not widen to complex.
v = ones(1, 8) * 2;
acc2 = 0;
for i = 1:n
    w = sqrt(v ./ k);
    acc2 = acc2 + sum(w);
end
assert(abs(acc2 - n * 8 * sqrt(2 / k)) < 1e-12);

disp('SUCCESS')
