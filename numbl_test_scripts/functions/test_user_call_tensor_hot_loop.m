% Hot-loop pattern: the outer loop repeatedly calls user functions that
% take and return tensors. The C-JIT emits each reachable callee as a
% static void in the same .c file, so the loop body becomes a chain of
% direct C calls with no JS boundary per iteration.
%
% Validates correctness of the tensor-arg + tensor-return ABI in the
% shape of code this was actually built for.

% 1D random-walk: each step takes the current state vector, produces a
% new vector one element longer (range slice + scalar growth), and the
% outer loop accumulates the final sum.
state = zeros(1, 3);
for k = 1:3
    state(k) = k;
end

total = 0;
for i = 1:5
    state = append_one(state, i * 10);
    total = total + sum(state);
end
% After each iteration:
% i=1: state=[1 2 3 10], sum=16
% i=2: state=[1 2 3 10 20], sum=36
% i=3: state=[1 2 3 10 20 30], sum=66
% i=4: state=[1 2 3 10 20 30 40], sum=106
% i=5: state=[1 2 3 10 20 30 40 50], sum=156
% total = 16+36+66+106+156 = 380
assert(total == 380)

% Pure-function pattern: tensor in, tensor out, no aliasing.
v = [1 2 3 4 5];
w = scale_by(v, 10);
assert(sum(w) == 150)
% Caller's v must not be mutated by the callee's writes on its local param copy.
assert(isequal(v, [1 2 3 4 5]))

% Matrix pass-through: outer doesn't index M, but callee reads M(i,j).
% Exercises the shape propagation (caller's maxIndexDim bumped so d0 is
% plumbed through the outer's ABI).
M = zeros(3, 4);
for r = 1:3
    for c = 1:4
        M(r, c) = r * 10 + c;
    end
end
row_sum = sum_row(M, 2);
assert(row_sum == (21 + 22 + 23 + 24))

disp('SUCCESS')

function w = append_one(v, x)
    w = zeros(1, length(v) + 1);
    for k = 1:length(v)
        w(k) = v(k);
    end
    w(length(v) + 1) = x;
end

function w = scale_by(v, s)
    w = zeros(1, length(v));
    for k = 1:length(v)
        w(k) = v(k) * s;
    end
end

function s = sum_row(M, r)
    s = 0;
    for c = 1:4
        s = s + M(r, c);
    end
end
