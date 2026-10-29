% More interpreter/JIT divergences found by the per-builtin conformance
% test (src/__tests__/jit-builtin-conformance.test.ts). Each check runs at
% the top level and inside a function, which the JIT compiles at --opt 1
% and --opt 2.

% min/max along a dim of a column (heap corruption at --opt 2).
assert(max([0.5; -3; 2], [], 1) == 2);
assert(max_dim1_k([0.5; -3; 2]) == 2 && min_dim1_k([0.5; -3; 2]) == -3);

% A bracketed single output of a builtin.
assert(bracket_k(2.5) == 2.5);
assert(isequal(bracket_k([1 2 3]), [1 3 6]));

% Shape builtins and min/max keep the logical class.
L = [true false true];
assert(islogical(flip(L)) && islogical(reshape(L, 3, 1)));
assert(islogical(repmat(L, 2, 1)) && islogical(diag(L)));
assert(islogical(cat(2, L, L)) && islogical(cummax(L)) && islogical(cummin(L)));
assert(islogical(max(L)) && islogical(min(L, [], 2)) && islogical(max(L, L)));
assert(~islogical(max(L, [0.5 2 1])));
[a, b, c, d] = logical_class_k(L);
assert(islogical(a) && islogical(b) && islogical(c) && islogical(d));

% A complex result whose imaginary parts are all zero is real.
assert(isreal(exp(complex(2, 0))) && isreal(sqrt_k([1 4 9])));
assert(isreal(triu([1+1i 2; -3i 4], 1)));
assert(isequal(sqrt_k([1 4 9]), [1 2 3]));

% logical of NaN raises.
assert(raises(@() logical(NaN), 'MATLAB:nologicalnan'));
assert(raises(@() logical_k([1 NaN]), 'MATLAB:nologicalnan'));
assert(raises(@() logical_k(NaN), 'MATLAB:nologicalnan'));
assert(isequal(logical_k([2 0 -1]), [true false true]));

% norm(v, +-Inf) of a vector holding NaN is NaN.
assert(isnan(norm([1 NaN 3], Inf)) && isnan(norm([1 NaN 3], -Inf)));
[p, q] = norm_inf_k([1 NaN 3]);
assert(isnan(p) && isnan(q));
[p, q] = norm_inf_k([1 -4 3]);
assert(p == 4 && q == 1);

% A complex number over zero divides each part: a NaN part stays NaN.
z = [1+2i, NaN+1i] ./ [2, 0];
assert(z(1) == 0.5+1i && isnan(real(z(2))) && imag(z(2)) == Inf);
z = rdivide_k([1+2i, NaN+1i], [2i, 0]);
assert(isnan(real(z(2))) && imag(z(2)) == Inf);

% Concatenating empties keeps their non-cat dims.
assert(isequal(size(cat(2, zeros(0, 3), zeros(0, 2))), [0 5]));
assert(isequal(size(cat_k(zeros(0, 3), zeros(0, 2))), [0 5]));
assert(isequal(size(cat_k([], zeros(0, 2))), [0 2]));
assert(isequal(size(cat_k([], [])), [0 0]));

% any of a complex array ignores NaN elements.
assert(isequal(any([1+2i, NaN+1i, 0], 1), [true false false]));
assert(isequal(any_dim1_k([1+2i, NaN+1i, 0]), [true false false]));

% mean of a complex vector with a NaN real part keeps its imaginary part.
m = mean_k([1+2i, -3i, 4, NaN+1i, -2-0.5i]);
assert(isnan(real(m)) && abs(imag(m) + 0.1) < 1e-15);

% diag of an empty vector.
assert(isequal(size(diag(zeros(1, 0))), [0 0]));
assert(isequal(size(diag([])), [0 0]));
assert(isequal(diag(zeros(0, 1), 2), zeros(2)));
assert(isequal(size(diag_k(zeros(1, 0))), [0 0]));

% printf keeps the sign of -0 and honors '+' for %e, %f and %g.
assert(strcmp(sprintf('%g|%+.3e|%f|%e', -0, -0, -0, -0), ...
  '-0|-0.000e+00|-0.000000|-0.000000e+00'));
assert(strcmp(sprintf('%+g|%+e|%+f|%06.2f', 1.5, 2, 3, -2.5), ...
  '+1.5|+2.000000e+00|+3.000000|-02.50'));
assert(strcmp(sprintf_k(-0), '-0|-0.000e+00|-0.000000'));

% not of a complex value raises.
assert(raises(@() ~complex(1, 2), 'MATLAB:Not:operandsNotReal'));

disp('SUCCESS');

function ok = raises(f, id)
ok = false;
try
  f();
catch e
  ok = strcmp(e.identifier, id);
end
end

function y = max_dim1_k(a)
y = max(a, [], 1);
end

function y = min_dim1_k(a)
y = min(a, [], 1);
end

function y = bracket_k(x)
[y] = cumsum(x);
end

function [a, b, c, d] = logical_class_k(L)
a = flip(L);
b = reshape(L, 3, 1);
c = max(L);
d = cummax(L);
end

function y = sqrt_k(x)
y = sqrt(x);
end

function y = logical_k(x)
y = logical(x);
end

function [p, q] = norm_inf_k(v)
p = norm(v, Inf);
q = norm(v, -Inf);
end

function z = rdivide_k(a, b)
z = a ./ b;
end

function y = cat_k(a, b)
y = cat(2, a, b);
end

function y = any_dim1_k(x)
y = any(x, 1);
end

function m = mean_k(x)
m = mean(x);
end

function y = diag_k(x)
y = diag(x);
end

function s = sprintf_k(x)
s = sprintf('%g|%+.3e|%f', x, x, x);
end
