% Stage 10 — function-call form of `and(a,b)` / `or(a,b)` / `not(a)`
% inside a JIT loop. Mirrors the chunkie pattern
%   `while(and(is > 0, ntry <= nnodes), ...)`
% which uses the function-call form rather than the `&&` operator. The
% JIT lowering folds these to `Binary(AndAnd|OrOr)` / `Unary(Not)` so
% the inner loop avoids the `$h.ib_and(...)` helper hop.
%
% ``%!numbl:assert_jit`` is placed inside each outer loop body to
% assert the surrounding loop got JIT-compiled — if the marker call
% survives to the interpreter (because lowering bailed), it throws.

% 1) Basic `and(a, b)` of scalar comparisons in an if-condition
total_and = 0;
for i = 1:200
    %!numbl:assert_jit
    a = mod(i, 7);
    b = mod(i, 11);
    if and(a > 2, b < 8)
        total_and = total_and + 1;
    end
end
% Verify against the operator form
expected_and = 0;
for i = 1:200
    a = mod(i, 7);
    b = mod(i, 11);
    if (a > 2) && (b < 8)
        expected_and = expected_and + 1;
    end
end
assert(total_and == expected_and, '1: and() vs && mismatch');

% 2) Basic `or(a, b)` of scalar comparisons
total_or = 0;
for i = 1:200
    %!numbl:assert_jit
    a = mod(i, 7);
    b = mod(i, 11);
    if or(a == 0, b == 0)
        total_or = total_or + 1;
    end
end
expected_or = 0;
for i = 1:200
    a = mod(i, 7);
    b = mod(i, 11);
    if (a == 0) || (b == 0)
        expected_or = expected_or + 1;
    end
end
assert(total_or == expected_or, '2: or() vs || mismatch');

% 3) `not(a)` of a scalar comparison
total_not = 0;
for i = 1:200
    %!numbl:assert_jit
    c = mod(i, 5);
    if not(c == 0)
        total_not = total_not + 1;
    end
end
% Equivalent: c != 0 → 200 - (count of c == 0)
expected_not = 0;
for i = 1:200
    c = mod(i, 5);
    if c ~= 0
        expected_not = expected_not + 1;
    end
end
assert(total_not == expected_not, '3: not() vs ~ mismatch');

% 4) Nested mix: and(or(...), not(...))
total_mix = 0;
for i = 1:200
    %!numbl:assert_jit
    a = mod(i, 4);
    b = mod(i, 6);
    if and(or(a == 0, b == 0), not(a == b))
        total_mix = total_mix + 1;
    end
end
expected_mix = 0;
for i = 1:200
    a = mod(i, 4);
    b = mod(i, 6);
    if ((a == 0) || (b == 0)) && (~(a == b))
        expected_mix = expected_mix + 1;
    end
end
assert(total_mix == expected_mix, '4: nested and(or, not) mismatch');

% 5) `while` with function-call `and` — direct mirror of chunkie's
%    `while(and(is > 0, ntry <= nnodes))`.
is = 5;
ntry = 0;
nnodes = 100;
loops = 0;
while and(is > 0, ntry <= nnodes)
    %!numbl:assert_jit
    is = is - 1;
    ntry = ntry + 1;
    loops = loops + 1;
end
assert(loops == 5, '5: while-and loop count');
assert(is == 0, '5: is decremented');
assert(ntry == 5, '5: ntry incremented');

disp('SUCCESS');
