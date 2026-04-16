% Verify that the JIT matches the interpreter (and MATLAB) for cases where
% the JIT's fast-path codegen is known to have diverged.
%
% Each helper is called directly (not inside a for-loop) three times so the
% function-call JIT specializes on the observed argument types and emits the
% fast-path code we care about exercising.

% ── Bug 1: asin(x) / acos(x) outside [-1,1] should be complex ──────────
y = do_asin(2); y = do_asin(2); y = do_asin(2);
assert(~isreal(y), 'asin(2) should be complex');
assert(abs(imag(y)) > 1, 'asin(2) imag part should be ~-1.3170');

y = do_acos(-1.5); y = do_acos(-1.5); y = do_acos(-1.5);
assert(~isreal(y), 'acos(-1.5) should be complex');
assert(abs(imag(y)) > 0.5, 'acos(-1.5) imag part should be ~-0.9624');

% ── Bug 2: `^` / `.^` with negative base and fractional exponent ───────
y = do_pow(-2, 0.5); y = do_pow(-2, 0.5); y = do_pow(-2, 0.5);
assert(~isreal(y), '(-2)^0.5 should be complex');
assert(abs(imag(y) - sqrt(2)) < 1e-10, '(-2)^0.5 should be sqrt(2)i');

v = do_elem_pow([-4; -1; 4; 9], 0.5);
v = do_elem_pow([-4; -1; 4; 9], 0.5);
v = do_elem_pow([-4; -1; 4; 9], 0.5);
assert(~isreal(v), '[-4;-1;4;9].^0.5 should be complex');
assert(abs(imag(v(1)) - 2) < 1e-10, '(-4).^0.5 should have imag ~ 2');
assert(abs(imag(v(2)) - 1) < 1e-10, '(-1).^0.5 should have imag ~ 1');
assert(abs(v(3) - 2) < 1e-10, '4.^0.5 should be 2');
assert(abs(v(4) - 3) < 1e-10, '9.^0.5 should be 3');

% ── Bug 3: scalar comparison returns class `logical` ───────────────────
y = do_lt(3, 5); y = do_lt(3, 5); y = do_lt(3, 5);
assert(islogical(y), '3 < 5 should be logical');

y = do_eq(3, 3); y = do_eq(3, 3); y = do_eq(3, 3);
assert(islogical(y), '3 == 3 should be logical');

y = do_gt(7, 5); y = do_gt(7, 5); y = do_gt(7, 5);
assert(islogical(y), '7 > 5 should be logical');

% ── Bug 4: `~x` with a logical operand ─────────────────────────────────
y = do_not(false); y = do_not(false); y = do_not(false);
assert(y == 1, '~false should be 1');
assert(islogical(y), '~false should be logical');

y = do_not(true); y = do_not(true); y = do_not(true);
assert(y == 0, '~true should be 0');

y = do_not_plus(false); y = do_not_plus(false); y = do_not_plus(false);
assert(y == 101, '~false + 100 should be 101');

% ── Bug 5: `==` / `~=` between a logical and a numeric literal ─────────
y = do_eq_zero(false); y = do_eq_zero(false); y = do_eq_zero(false);
assert(y == 1, 'false == 0 should be 1');

y = do_ne_zero(false); y = do_ne_zero(false); y = do_ne_zero(false);
assert(y == 0, 'false ~= 0 should be 0');

% ── Bug 6: logical indexing `v(mask)` inside a JIT'd function ──────────
r = do_logical_index([1, -2, 3, -4, 5]);
r = do_logical_index([1, -2, 3, -4, 5]);
r = do_logical_index([1, -2, 3, -4, 5]);
assert(isequal(r, [1, 3, 5]), 'logical indexing should return [1 3 5]');

% ── Bug 7: `&&` / `||` with logical operands ───────────────────────────
y = do_and(false, true); y = do_and(false, true); y = do_and(false, true);
assert(y == 0, 'false && true should be 0');

y = do_and(false, false); y = do_and(false, false); y = do_and(false, false);
assert(y == 0, 'false && false should be 0');

y = do_or(false, false); y = do_or(false, false); y = do_or(false, false);
assert(y == 0, 'false || false should be 0');

y = do_or(true, false); y = do_or(true, false); y = do_or(true, false);
assert(y == 1, 'true || false should be 1');

disp('SUCCESS')

% ── Helpers (each wraps the op in its own function to trigger JIT specialization) ──
function r = do_asin(x); r = asin(x); end
function r = do_acos(x); r = acos(x); end
function r = do_pow(a, b); r = a ^ b; end
function r = do_elem_pow(a, b); r = a .^ b; end
function r = do_lt(a, b); r = a < b; end
function r = do_eq(a, b); r = a == b; end
function r = do_gt(a, b); r = a > b; end
function r = do_not(x); r = ~x; end
function r = do_not_plus(x); r = ~x + 100; end
function r = do_eq_zero(x); r = x == 0; end
function r = do_ne_zero(x); r = x ~= 0; end
function r = do_logical_index(v); m = v > 0; r = v(m); end
function r = do_and(a, b); r = a && b; end
function r = do_or(a, b); r = a || b; end
