% Tensor arguments and returns crossing the C-JIT call boundary.
%
% Each callee carries `%!numbl:assert_jit c`, so at --opt 2 a call that
% does not run as C (e.g. a marshaling failure that bails) reaches the
% interpreter and raises. The script body itself stays interpreted
% (complex-tensor `isequal` is not JIT-compilable), so every call goes
% through the cjit-call executor and marshals its tensors.

a = [1 2 3 4 5];
y = mul_add(a, a);
assert(isequal(y, a .* a + 1), 'real row tensor');
assert(isequal(a, [1 2 3 4 5]), 'caller tensor unchanged');

M = reshape(1:12, 3, 4);
y = mul_add(M, M);
assert(isequal(size(y), [3 4]) && isequal(y, M .* M + 1), 'real matrix');

T = reshape(1:24, 2, 3, 4);
y = mul_add(T, 2 * T);
assert(isequal(size(y), [2 3 4]) && isequal(y, 2 * T .* T + 1), '3-D tensor');

z = [1+2i, 3-1i, -2i];
y = mul_add(z, z);
assert(isequal(y, z .* z + 1), 'complex tensor');

% Real value into a param the spec types as complex (the second arg
% widens to complex after the first call).
w1 = scale2(z, z);
w2 = scale2(z, [1 2 3]);
assert(isequal(w1, z .* z) && isequal(w2, z .* [1 2 3]), 'real into complex');

[s, d] = sum_diff([1 2 3], [4 5 6]);
assert(isequal(s, [5 7 9]) && isequal(d, [-3 -3 -3]), 'multi-output tensors');

total = 0;
for k = 1:50
    total = total + sum(mul_add(a, a));
end
assert(total == 50 * sum(a .* a + 1), 'repeated calls');

% Two loops in a row: the first loop's synthetic spec (whose pseudo-output
% `j` is unassigned on a zero-trip run) stays in the shared spec map and
% must not make the second loop's C compile decline.
acc = zeros(1, 5);
nt = total / total + 2; % 3, not a compile-time constant
for j = 1:nt
    %!numbl:assert_jit c
    acc = acc + a;
end
for i = 1:nt - 1
    %!numbl:assert_jit c
    acc = acc .* 2;
end
assert(isequal(acc, 12 * a), 'consecutive loops');

disp('SUCCESS')

function y = mul_add(a, b)
    %!numbl:assert_jit c
    y = a .* b + 1;
end

function y = scale2(a, b)
    %!numbl:assert_jit c
    y = a .* b;
end

function [s, d] = sum_diff(a, b)
    %!numbl:assert_jit c
    s = a + b;
    d = a - b;
end
