% Test: transpose (.') and ctranspose (') on complex scalars under JIT

z = 1 + 2i;

% Non-conjugate transpose of a scalar is the scalar itself
a = z.';
assert(real(a) == 1);
assert(imag(a) == 2);

% Conjugate transpose of a scalar is the conjugate
b = z';
assert(real(b) == 1);
assert(imag(b) == -2);

% Exercise inside a function so it actually gets JITed
function [u, v] = inner(w)
  u = w.';
  v = w';
end

[c, d] = inner(3 + 4i);
assert(real(c) == 3 && imag(c) == 4);
assert(real(d) == 3 && imag(d) == -4);

% Real scalar: both forms are identity
r = 7;
assert(r.' == 7);
assert(r' == 7);

disp('SUCCESS')
