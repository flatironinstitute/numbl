% Divergences between the interpreter and the JIT found by the per-builtin
% conformance test (src/__tests__/jit-builtin-conformance.test.ts). Each
% check runs once at the top level and once inside a function, which the
% JIT compiles at --opt 1 and --opt 2.

% sort is stable and treats -0 and 0 as equal.
s = sort([3 -0 0 NaN -Inf 2 NaN 0 -0]);
assert(isequal(1 ./ s(2:5), [-Inf Inf Inf -Inf]));
s = sort_k([1 -2 3.5 0 -0 NaN Inf -Inf]);
assert(isequal(1 ./ s(3:4), [Inf -Inf]));

% any ignores NaN, along every dimension; all counts it as nonzero.
assert(~any(NaN) && ~any(complex(NaN, 1)) && all(NaN));
assert(isequal(any([1 NaN 0], 1), [true false false]));
assert(isequal(any([NaN 0], 3), [false false]));
assert(isequal(any_k(NaN), false));
assert(isequal(any_dim1_k([1 NaN 0]), [true false false]));
assert(isequal(any_dim3_k([NaN 0; 1 NaN]), logical([0 0; 1 0])));

% Reductions of an empty matrix run along its first non-unit dimension.
assert(isequal(size(sum(zeros(0, 3))), [1 3]));
assert(isequal(sum_k(zeros(0, 3)), [0 0 0]));
assert(isequal(prod_k(zeros(0, 3)), [1 1 1]));
assert(isequal(size(mean_k(zeros(0, 3))), [1 3]));
assert(isequal(any_k(zeros(0, 3)), false(1, 3)));

% sign(-0) is 0, not -0.
assert(1 / sign(-0) == Inf);
assert(isequal(1 ./ sign([-0 0]), [Inf Inf]));
assert(isequal(1 ./ sign_k([-0 0]), [Inf Inf]));

% imag of a real array is +0 everywhere, even for NaN and Inf.
assert(isequal(1 ./ imag_k([1 -2 NaN Inf]), [Inf Inf Inf Inf]));

% angle of a real -0 is 0.
assert(angle(-0) == 0 && angle_k(-0) == 0 && angle_k(-2) == pi);

% linspace and logspace round the point count down.
assert(isequal(linspace(0, 1, 2.5), [0 1]));
assert(isequal(linspace_k(0, 1), [0 1]));
assert(numel(logspace_k(0, 1)) == 2);

% A string or a struct is a 1x1 value and is not real.
st = struct('a', 1);
assert(isrow("abc") && iscolumn("abc") && isscalar(st) && isrow(st));
assert(~isreal("abc") && ~isreal(st) && isreal('abc'));
assert(isequal(typequery_k("abc"), [true true true false]));
assert(isequal(typequery_k(st), [true true true false]));

% double keeps a complex value complex, unless its imaginary parts are
% all zero.
z = complex(2, 0);
assert(isreal(double(z)) && isreal(double(complex([1 2], [0 0]))));
assert(double_k(complex(1.5, -2)) == complex(1.5, -2));

% max/min of a complex and a real operand compare magnitudes.
assert(max(z, -1.5) == 2 && max_k(z, -1.5) == 2);
assert(min_k(complex(1.5, -2), 2.5) == complex(1.5, -2));

% x / B with a non-scalar B is not element-wise division.
assert(raises_k(2.5, [2 -1; 0.5 3]));

% Complex sort orders by magnitude, then angle, with NaN last.
c = sort_k([1+2i, -3i, 4, NaN+1i, -2-0.5i]);
assert(isequal(c(1:4), [-2-0.5i, 1+2i, -3i, 4]) && isnan(real(c(5))));

% not rejects a complex operand.
assert(not_raises_k(complex(1, 2)));

disp('SUCCESS')

function y = sort_k(x)
y = sort(x);
end
function y = any_k(x)
y = any(x);
end
function y = any_dim1_k(x)
y = any(x, 1);
end
function y = any_dim3_k(x)
y = any(x, 3);
end
function y = sum_k(x)
y = sum(x);
end
function y = prod_k(x)
y = prod(x);
end
function y = mean_k(x)
y = mean(x);
end
function y = sign_k(x)
y = sign(x);
end
function y = imag_k(x)
y = imag(x);
end
function y = angle_k(x)
y = angle(x);
end
function y = linspace_k(a, b)
y = linspace(a, b, 2.5);
end
function y = logspace_k(a, b)
y = logspace(a, b, 2.5);
end
function y = typequery_k(x)
y = [isrow(x), iscolumn(x), isscalar(x), isreal(x)];
end
function y = double_k(x)
y = double(x);
end
function y = max_k(a, b)
y = max(a, b);
end
function y = min_k(a, b)
y = min(a, b);
end
function r = raises_k(a, b)
r = false;
try
  y = a / b; %#ok<NASGU>
catch
  r = true;
end
end
function r = not_raises_k(z)
r = false;
try
  y = not(z); %#ok<NASGU>
catch
  r = true;
end
end
