% Namespace function calls inside a JIT loop — the chunkie `oneintp`
% pattern, where the inner workhorse calls package functions like
%   chnk.perp(dint)
%   chnk.flagself(...)
% which the parser produces as
%   MethodCall(base=Ident("chnk"), name="perp", args=[dint])
% (any postfix `.name(...)` on an expression is a MethodCall in our AST.)
%
% Today the JIT lowers MethodCall only for the chained struct-array
% form `T.nodes(i).leaf` and bails on every other shape. That kills
% JIT for chunkie's hottest functions because `chnk.perp` is reached
% from inside `oneintp`'s lowering, taking the whole adapgausskerneval
% inner loop down with it.
%
% Uses the +mymath package already in this directory for resolution.

%!numbl:assert_jit c
n = 100;

% 1) Namespace call with scalar args — the simplest shape.
acc1 = 0;
for i = 1:n
    %!numbl:assert_jit
    acc1 = acc1 + mymath.add_two(i, i * 2);
end

% 2) Namespace call where args are loop-local computed values (more
%    representative of `chnk.perp(dint)` where `dint = ds*interpmat`
%    is computed in-loop).
acc2 = 0;
for i = 1:n
    %!numbl:assert_jit
    a = i * 1.5;
    b = a + 3.0;
    acc2 = acc2 + mymath.add_two(a, b);
end

% Correctness checks (run-mode independent).
expected1 = 0;
for i = 1:n
    expected1 = expected1 + (i + i * 2);
end
assert(acc1 == expected1, '1: acc1 value');

expected2 = 0;
for i = 1:n
    a = i * 1.5;
    b = a + 3.0;
    expected2 = expected2 + (a + b);
end
assert(abs(acc2 - expected2) < 1e-9, '2: acc2 value');

disp('SUCCESS')
