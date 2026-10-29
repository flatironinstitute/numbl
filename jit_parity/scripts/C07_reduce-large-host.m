% Full reductions of inputs of 256 elements or more run through the
%   host's flat-reduce kernel at --opt 1 (the interpreter's own, native
%   when the addon is loaded) and through the C kernel at --opt 2. All
%   accumulate sequentially, so ill-conditioned sums must agree bit for
%   bit, as must the NaN-skipping min/max and the all-NaN case.
function r = f(v)
  r = [sum(v), mean(v), prod(v), min(v), max(v)];
  for k = 2:3
    r = [sum(v), mean(v), prod(v), min(v), max(v)];
  end
end
v = [1e16, ones(1, 2000), -1e16, 0.1 * (1:500)];
fprintf('%.20g\n', f(v));
w = 1 + 1e-3 * sin(1:3000);
fprintf('%.20g\n', f(w));
x = [NaN, -3 * ones(1, 1500), NaN, 7];
fprintf('%.20g\n', f(x));
fprintf('%.20g\n', f(NaN(1, 1200)));
