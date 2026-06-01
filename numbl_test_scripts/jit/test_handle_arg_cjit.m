% Function handles passed as CALL ARGUMENTS work through the JIT: when the
% call site is absorbed into a JIT-compiled region, the callee is
% specialized with the handle's concrete type and its body is inlined at
% the `f(x)` call. This is the chunkie `oneintp(kern, ...)` shape — a
% routine that takes a kernel handle and calls it in a loop.
%
% The `%!numbl:assert_jit c` inside `apply_sum` fails the test if the
% callee runs in the interpreter at --opt 2. Three distinct handles are
% passed to the same callee to exercise per-handle specialization (each
% concrete handle is a separate compiled spec).

x = (1:1000).';

% Named handle argument.
s1 = apply_sum(@sq, x);
% Anonymous capture-free handle argument.
s2 = apply_sum(@(t) t + 1, x);
% Reuse with a different named handle — must specialize separately, not
% reuse s1's compiled spec.
s3 = apply_sum(@neg, x);

assert(abs(s1 - sum(x.^2 + 1)) < 1e-6, '1: named handle arg');
assert(abs(s2 - sum((x + 1) + 1)) < 1e-6, '2: anon handle arg');
assert(abs(s3 - sum(-x + 1)) < 1e-6, '3: distinct named handle arg');

disp('SUCCESS')

% Callee: takes a handle and reduces it over the column vector in a loop.
function s = apply_sum(f, x)
    s = 0;
    for i = 1:numel(x)
        %!numbl:assert_jit c
        s = s + f(x(i)) + 1;
    end
end

function y = sq(x)
    y = x.^2;
end

function y = neg(x)
    y = -x;
end
