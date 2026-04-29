% Cell-list-unpack patterns inside JIT loops. The chunkie/chunkerfunc
% resolve loop uses `[out{1:nout}] = fcurve(ts)` plus dynamic `out{j}`
% reads/writes; today the JIT bails on all of them.
%
% Each %!numbl:assert_jit pins one shape. Cases 1-2 currently fail —
% they document the gap so the cell-on-LHS / cell-with-non-literal-
% index work (Option 2/3 in the e2 plan) can drive against them.

fcurve3 = @(t) deal(cos(t), -sin(t), -cos(t));
base_ts = linspace(0, 1, 32).';
[a0, b0, c0] = fcurve3(base_ts);  % warmup

n = 20;

% 1) Cell-list-unpack with literal range: the form chunkerfunc uses
%    after substituting the introspected nout. `lowerMultiAssign`
%    only supports plain Var lvalues today; cell-element lvalues
%    bail.
out = cell(3, 1);
acc1 = 0;
for i = 1:n
    %!numbl:assert_jit
    [out{1:3}] = fcurve3(base_ts);
    acc1 = acc1 + out{1}(1) + out{2}(1) + out{3}(1);
end

% 2) Cell-list-unpack with a Var range (chunkerfunc's literal
%    `[out{1:nout}] = fcurve(ts)` — nout is 3 throughout the spec but
%    is a Var in the source).
nout = 3;
out2 = cell(3, 1);
acc2 = 0;
for i = 1:n
    %!numbl:assert_jit
    [out2{1:nout}] = fcurve3(base_ts);
    acc2 = acc2 + out2{1}(1) + out2{2}(1) + out2{3}(1);
end

% 3) Cell-write with non-literal (loop-variable) index after a
%    cell-list-unpack. The chunkerfunc resolve loop has a small inner
%    `for j = nout+1:3; out{j} = out{j-1}*M*c; end` — the body is dead
%    when nout==3 but lowering still has to type-check it. Reads of
%    out{j-1} must resolve to a known type when every tracked
%    element shares one, so the Binary RHS lowers.
out3 = cell(3, 1);
M = ones(2, 2);
acc3 = 0;
for i = 1:n
    %!numbl:assert_jit
    [out3{1:3}] = fcurve3(base_ts);
    for j = 4:3  % statically empty range — body is dead but still has to lower
        out3{j} = out3{j-1} * M * 2;
    end
    acc3 = acc3 + out3{1}(1);
end

% Correctness check
expected = n * (cos(0) - sin(0) - cos(0));
assert(abs(acc1 - expected) < 1e-12, '1: acc1');
assert(abs(acc2 - expected) < 1e-12, '2: acc2');
assert(abs(acc3 - n * cos(0)) < 1e-12, '3: acc3');

disp('SUCCESS')
