% Matrix products large enough to leave the JIT's inline loop (they go to
% the host's LAPACK provider stack): real, complex, mixed real/complex, the
% 1xk * kx1 scalar form, and matrix * vector, checked against a plain loop.

A = reshape(1 ./ (1:1200), 40, 30);
B = reshape(1 ./ (3:1502), 30, 50);
x = 1 ./ (7:2006)';
Z = A(1:20, :) + 1i * A(21:40, :);
Y = B + 1i * B(:, end:-1:1);

[C, s, W, V, U, Q] = products(A, B, x, Z, Y);

assert(isequal(size(C), [40 50]));
assert(maxdiff(C, ref(A, B)) < 1e-14);
assert(isscalar(s) && abs(s - sum(x .^ 2)) < 1e-14);
assert(maxdiff(W, ref(Z, B)) < 1e-14);
assert(maxdiff(V, ref(A, Y)) < 1e-14);
assert(maxdiff(U, ref(Z, Y)) < 1e-14);
assert(isequal(size(Q), [40 1]));
assert(maxdiff(Q, ref(A, B(:, 1))) < 1e-14);

% A product whose imaginary part cancels is still a correct value.
R = products2(Z, conj(Z).');
assert(maxdiff(R, ref(Z, conj(Z).')) < 1e-14);

disp('SUCCESS');

function [C, s, W, V, U, Q] = products(A, B, x, Z, Y)
  C = A * B;
  s = x' * x;
  W = Z * B;
  V = A * Y;
  U = Z * Y;
  Q = A * B(:, 1);
end

function R = products2(P, Q)
  R = P * Q;
end

function d = maxdiff(X, Y)
  d = max(abs(X(:) - Y(:)));
end

function C = ref(A, B)
  Ar = real(A); Ai = imag(A); Br = real(B); Bi = imag(B);
  C = refReal(Ar, Br) - refReal(Ai, Bi) + 1i * (refReal(Ar, Bi) + refReal(Ai, Br));
end

function C = refReal(A, B)
  C = zeros(size(A, 1), size(B, 2));
  for i = 1:size(A, 1)
    for j = 1:size(B, 2)
      acc = 0;
      for p = 1:size(A, 2)
        acc = acc + A(i, p) * B(p, j);
      end
      C(i, j) = acc;
    end
  end
end
