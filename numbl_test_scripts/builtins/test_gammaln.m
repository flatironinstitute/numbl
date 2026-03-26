% Test gammaln builtin
tol = 1e-10;

% Basic values
assert(abs(gammaln(1) - 0) < tol);
assert(abs(gammaln(2) - 0) < tol);
assert(abs(gammaln(3) - log(2)) < tol);
assert(abs(gammaln(4) - log(6)) < tol);
assert(abs(gammaln(5) - log(24)) < tol);

% Fractional values
assert(abs(gammaln(0.5) - log(sqrt(pi))) < tol);
assert(abs(gammaln(1.5) - log(sqrt(pi)/2)) < tol);

% Large values (where gamma overflows but gammaln does not)
val = gammaln(200);
assert(val > 0 && isfinite(val));

% Edge cases
assert(gammaln(0) == Inf);
try
    gammaln(-1);
    assert(false);
catch e
    assert(contains(e.message, 'nonnegative'));
end

% Vector input
x = [1 2 3 4 5];
y = gammaln(x);
expected = [0 0 log(2) log(6) log(24)];
assert(all(abs(y - expected) < tol));

disp('SUCCESS');
