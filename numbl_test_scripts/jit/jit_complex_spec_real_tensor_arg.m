% A function or loop first run with a complex tensor input, then with a
% real one. Type widening reuses the complex specialization for the real
% input, so the real tensor must cross the JIT boundary with a zero
% imaginary part, as a real scalar does for a complex scalar parameter.

% call shape
y = dbl([1+2i; 3]);
assert(isequal(y, [2+4i; 6]));
y = dbl([1; 3]);
assert(isequal(y, [2; 6]) && isreal(y));

% loop shape (the loop inside the function sees p complex, then real)
p = recur([1i; 2]);
assert(isequal(p, [1i 2i 6i; 2 4 12]));
p = recur([1; 2]);
assert(isequal(p, [1 2 6; 2 4 12]) && isreal(p));

% a triple-loop matrix product: complex operands, then real ones
A = [1 2; 3 4];
B = [5; 6];
assert(isequal(matprod(A + 1i*A, B), [17+17i; 39+39i]));
C = matprod(A, B);
assert(isequal(C, [17; 39]) && isreal(C));

% a struct field widened from complex to real
s.v = [1i; 2];
assert(isequal(fieldsum(s), 2 + 1i));
s.v = [1; 2];
assert(isequal(fieldsum(s), 3));

disp('SUCCESS')

function y = dbl(x)
  y = x;
  for k = 1:numel(x)
    y(k) = 2*x(k);
  end
end

function p = recur(x)
  p = zeros(numel(x), 3);
  p(:, 1) = x;
  for k = 2:3
    p(:, k) = k * p(:, k-1);
  end
end

function C = matprod(A, B)
  C = zeros(size(A, 1), size(B, 2));
  for i = 1:size(A, 1)
    for j = 1:size(B, 2)
      acc = 0;
      for q = 1:size(A, 2)
        acc = acc + A(i, q) * B(q, j);
      end
      C(i, j) = acc;
    end
  end
end

function t = fieldsum(s)
  t = 0;
  for k = 1:numel(s.v)
    t = t + s.v(k);
  end
end
