% Value-sensitive builtins (isreal / min / max / sort) must treat a
% complex-typed tensor whose imaginary lane is entirely zero as real:
% order by value (not magnitude) and report isreal == true. This matches
% the interpreter (--opt 0), the JIT (--opt 1/2), and MATLAB on real data.
% The JIT routinely produces such tensors when it cannot prove realness
% at compile time (e.g. sqrt of a value whose sign isn't statically known).

% --- complex tensor, all imaginary parts zero ---
zr = complex([-1 -0.577 0.3 0.9], [0 0 0 0]);
assert(isreal(zr) == 1);              % all-zero imag -> real
assert(min(zr) == -1);                % value order, not |.|
assert(max(zr) == 0.9);
s = sort(zr);
assert(s(1) == -1 && s(2) == -0.577 && s(3) == 0.3 && s(4) == 0.9);
sd = sort(zr, 'descend');
assert(sd(1) == 0.9 && sd(2) == 0.3 && sd(3) == -0.577 && sd(4) == -1);

% --- lifted sqrt: argument's sign not statically provable, but every
%     value is positive, so the result is real-valued ---
y = [0.2 0.5 0.9 -0.3 0.7];
w = sqrt(1 - y.^2/2);
assert(isreal(w) == 1);
assert(abs(min(w) - sqrt(0.595)) < 1e-12);   % smallest value (y = 0.9)
assert(abs(max(w) - sqrt(0.98)) < 1e-12);     % largest value  (y = 0.2)

% --- min / max along a dimension on an all-zero-imag matrix ---
M = complex([2 -5; -3 1], zeros(2, 2));
cmin = min(M, [], 1);                 % column minima by value
assert(cmin(1) == -3 && cmin(2) == -5);
rmax = max(M, [], 2);                 % row maxima by value
assert(rmax(1) == 2 && rmax(2) == 1);

% --- regression guard: genuinely complex data (nonzero imag) keeps
%     complex/magnitude semantics ---
zc = complex([3 4], [4 0]);           % magnitudes 5 and 4
assert(isreal(zc) == 0);
mn = min(zc);                         % min magnitude -> 4 + 0i
assert(real(mn) == 4 && imag(mn) == 0);
mx = max(zc);                         % max magnitude -> 3 + 4i
assert(real(mx) == 3 && imag(mx) == 4);

fprintf('SUCCESS\n');
