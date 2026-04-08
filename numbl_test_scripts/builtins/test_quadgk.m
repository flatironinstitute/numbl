% Test quadgk — adaptive Gauss-Kronrod 7-15 quadrature.

% --- Smooth integrands with known analytic values ---

% int_0^pi sin(x) dx = 2
r = quadgk(@sin, 0, pi);
assert(abs(r - 2) < 1e-10, sprintf('sin: got %.15g', r));

% int_0^1 x^3 dx = 1/4
r = quadgk(@(x) x.^3, 0, 1);
assert(abs(r - 0.25) < 1e-12, sprintf('x^3: got %.15g', r));

% int_0^1 1/(1+x^2) dx = pi/4
r = quadgk(@(x) 1 ./ (1 + x.^2), 0, 1);
assert(abs(r - pi / 4) < 1e-10, sprintf('1/(1+x^2): got %.15g', r));

% int_-5^5 exp(-x^2) dx ≈ sqrt(pi)
r = quadgk(@(x) exp(-x.^2), -5, 5);
assert(abs(r - sqrt(pi)) < 1e-8, sprintf('exp(-x^2): got %.15g', r));

% int_-1^1 (1 - x.^2) dx = 4/3
r = quadgk(@(x) 1 - x.^2, -1, 1);
assert(abs(r - 4/3) < 1e-12, sprintf('1-x^2: got %.15g', r));

% --- Reversed limits: int_b^a = -int_a^b ---
r1 = quadgk(@sin, 0, pi);
r2 = quadgk(@sin, pi, 0);
assert(abs(r1 + r2) < 1e-12, 'reversed limits');

% --- Degenerate interval ---
assert(quadgk(@sin, 1.5, 1.5) == 0, 'degenerate interval');

% --- Closure capturing a parameter ---
k = 2.5;
r = quadgk(@(x) sin(k * x), 0, pi / k);
assert(abs(r - 2 / k) < 1e-10, sprintf('closure: got %.15g', r));

% --- Name-value options ---
r = quadgk(@(x) exp(x), 0, 1, 'RelTol', 1e-12, 'AbsTol', 1e-14);
assert(abs(r - (exp(1) - 1)) < 1e-11, sprintf('opts: got %.15g', r));

% --- Two-output form: [q, errbnd] = quadgk(...) ---
[q, errbnd] = quadgk(@(x) x.^2, 0, 2);
assert(abs(q - 8/3) < 1e-12, sprintf('two-output q: %.15g', q));
assert(errbnd >= 0, 'errbnd must be non-negative');
assert(errbnd < 1e-6, sprintf('errbnd too large: %g', errbnd));

% --- Vector-accepting integrand returning a vector ---
% (This is the MATLAB contract — quadgk passes a vector of nodes.)
calls = 0;
fun = @(x) track_and_return(x);
r = quadgk(fun, 0, 1);
assert(abs(r - 1/3) < 1e-12, sprintf('vector integrand: %.15g', r));

disp('SUCCESS');

function y = track_and_return(x)
    % Sanity: quadgk should call with a vector, not a scalar.
    assert(numel(x) > 1, 'integrand received a scalar, expected vector');
    y = x.^2;
end
