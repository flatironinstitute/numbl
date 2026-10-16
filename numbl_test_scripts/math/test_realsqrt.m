% realsqrt: real square root that errors for negative or complex input

assert(realsqrt(4) == 2);
assert(realsqrt(0) == 0);
assert(isnan(realsqrt(NaN)));
assert(realsqrt(Inf) == Inf);
v = realsqrt([1 4 9]);
assert(isequal(v, [1 2 3]));
assert(isreal(v));

% An input whose sign is not known statically stays real, in a loop
% and through a function, with scalar and tensor arguments.
s = 0;
for k = 1:200
  s = s + realsqrt(sin(k)^2);
end
assert(isreal(s));
assert(abs(s - sum(abs(sin(1:200)))) < 1e-10);
w = rs_helper(abs(cos(1:5)));
assert(isreal(w));
assert(max(abs(w - realsqrt(abs(cos(1:5))))) < 1e-15);

% Negative or complex input is an error.
rs_expect_error(@() realsqrt(-1));
rs_expect_error(@() realsqrt([1 -4 9]));
rs_expect_error(@() realsqrt(1i));
for k = 1:3
  rs_expect_error(@() rs_helper(-k * [1 2 3]));
end

disp('SUCCESS');

function y = rs_helper(x)
  y = realsqrt(x .* x .* x) ./ abs(x);
end

function rs_expect_error(f)
  threw = false;
  try
    f();
  catch err
    threw = true;
    assert(contains(err.message, 'Realsqrt produced complex result'));
  end
  assert(threw);
end
