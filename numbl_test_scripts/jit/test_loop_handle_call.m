% Loops that call function handles need to JIT for the chunkerfunc-style
% workloads (chunkie discretize phase). Each `%!numbl:assert_jit` here
% pins one shape that should lower at --opt 1; if any of them fall back
% to the interpreter, the test fails so we notice the regression / gap.
%
% Status: cases 1 and 2 currently pass (single-output handle, simple
% args). Cases 3+ are EXPECTED-FAIL until we extend the JIT — they pin
% the gap so we can drive each blocker to fix.

% --- Setup: register two simple handles. The 1-output handle is the
% baseline, the 3-output one mirrors chunkie's `fcurve = @(t)
% starfish(t,...)` shape.
fcurve1 = @(t) cos(t);
fcurve3 = @(t) deal(cos(t), -sin(t), -cos(t));

base_ts = linspace(0, 1, 32).';
% Warm both handles so probe + first call don't pollute timing.
r0 = fcurve1(base_ts);
[a0, b0, c0] = fcurve3(base_ts);

n = 20;

% 1) Single-output handle, Var arg from outer scope. Currently JITs.
acc1 = 0;
for i = 1:n
    %!numbl:assert_jit
    r = fcurve1(base_ts);
    acc1 = acc1 + r(1);
end

% 2) Single-output handle, Var arg built inside the loop body. Should
%    JIT (the probe synthesizes a representative tensor from the JIT
%    type) but currently bails — the probe runs at JIT-compile time
%    when the loop-local var hasn't been bound yet.
acc2 = 0;
for i = 1:n
    %!numbl:assert_jit
    ts = base_ts + i;
    r = fcurve1(ts);
    acc2 = acc2 + r(1);
end

% 3) Single-output handle, composite expression as arg. Even smaller
%    surface — this currently bails because the probe only accepts Var
%    / NumberLiteral lowered args.
acc3 = 0;
for i = 1:n
    %!numbl:assert_jit
    r = fcurve1(base_ts + i);
    acc3 = acc3 + r(1);
end

% 4) Multi-output handle call, the chunkerfunc/fcurve(ts) shape.
%    `lowerMultiAssign` only accepts IBuiltin RHS today (jitLowerStmt.ts
%    "user function multi-output not yet supported"), so this bails.
acc4 = 0;
for i = 1:n
    %!numbl:assert_jit
    [aa, bb, cc] = fcurve3(base_ts);
    acc4 = acc4 + aa(1) + bb(1) + cc(1);
end

% Correctness self-check (run-mode independent — these should match
% irrespective of whether JIT lowered or the interpreter ran).
assert(abs(acc1 - n * cos(0)) < 1e-12, '1: acc1 value');
assert(abs(acc2 - sum(cos((1:n).'))) < 1e-12, '2: acc2 value');
assert(abs(acc3 - sum(cos((1:n).'))) < 1e-12, '3: acc3 value');
assert(abs(acc4 - n * (cos(0) - sin(0) - cos(0))) < 1e-12, '4: acc4 value');

disp('SUCCESS')
