% Element-wise kernels (fused loops) and fused full reductions must give
% exactly the values of the same expression evaluated one operation at a
% time: NaN, Inf and signed zero included, and reductions summed in
% linear order. Covers static shapes, dynamic shapes that agree, dynamic
% shapes that broadcast (the kernel's guard falls back), and shapes that
% do not agree (the same error).

base = [-2.5 -1.5 -0.5 -0 0 0.5 1.5 2.5 NaN Inf -Inf 3.7 -3.7 1e-300];
for reps = [1 300]
    x = repmat(base, 1, reps)';
    y = x(end:-1:1);
    w = 0.25 * x + 1;

    [p, q, r] = fused(x, y, w, 0.5);
    t1 = x .* y;
    t2 = w .* 0.5;
    t3 = t1 + t2;
    pref = t3 - x;
    assert(same(p, pref), 'arith chain');
    t4 = abs(x - y);
    t5 = floor(w);
    t6 = sqrt(abs(y) + 1);
    qref = t4 + t5 ./ t6;
    assert(same(q, qref), 'exact unary chain');
    rref = -x .* 2 - y ./ 4;
    assert(same(r, rref), 'unary minus');

    [s1, s2, s3] = reduced(x, y);
    u = x .* 0.5 - y;
    assert(same(s1, sum(u)), 'sum of a fused tree');
    v = abs(x) .* 0.5 + 1;
    assert(same(s2, prod(v)), 'prod of a fused tree');
    z = x - 2 .* y;
    assert(same(s3, mean(z)), 'mean of a fused tree');

    finite = x(isfinite(x));
    [f1, f2] = reduced(finite, finite(end:-1:1));
    uf = finite .* 0.5 - finite(end:-1:1);
    assert(same(f1, sum(uf)), 'finite sum');
    assert(isfinite(f1), 'finite sum is finite');
    assert(same(f2, prod(abs(finite) .* 0.5 + 1)), 'finite prod');
end

% Signed zero survives a kernel.
nz = negzero([0; 0; 0]);
assert(all(1 ./ nz == -Inf), 'signed zero');

% Shapes known only at run time: equal, broadcasting, and mismatched.
assert(isequal(dynamic(ones(3, 1), 2 * ones(3, 1)), 3 * ones(3, 1) + 1), ...
    'dynamic equal shapes');
assert(isequal(dynamic(ones(3, 1), [1 2]), [3 4; 3 4; 3 4]), ...
    'dynamic broadcast');
try
    dynamic(ones(3, 1), ones(2, 1));
    error('expected a dimension error');
catch err
    assert(~isempty(strfind(err.message, 'agree')) || ...
        ~isempty(strfind(err.message, 'compatible')), err.message);
end

disp('SUCCESS')

function [p, q, r] = fused(x, y, w, k)
    %!numbl:assert_jit
    p = x .* y + w .* k - x;
    q = abs(x - y) + floor(w) ./ sqrt(abs(y) + 1);
    r = -x .* 2 - y ./ 4;
end

function [s1, s2, s3] = reduced(x, y)
    %!numbl:assert_jit
    s1 = sum(x .* 0.5 - y);
    s2 = prod(abs(x) .* 0.5 + 1);
    s3 = mean(x - 2 .* y);
end

function z = negzero(x)
    %!numbl:assert_jit
    z = -x .* 1 - 0;
end

function z = dynamic(a, b)
    z = a + b .* 1 + 1;
end

function t = same(a, b)
    t = isequal(size(a), size(b)) && ...
        all((a(:) == b(:) & (a(:) ~= 0 | 1 ./ a(:) == 1 ./ b(:))) | ...
            (isnan(a(:)) & isnan(b(:))));
end
