% R: indexing and indexed-store errors raised inside JIT'd code, with the
% interpreter's wording (runtime/indexPlan.ts): invalid and non-integer
% indices, per-position bounds, logical masks, and slice stores whose RHS
% does not fit (element count for linear stores, non-singleton dimensions
% for subscripted ones).
for c = 1:10
  try
    r = trigger(c, 3);
    fprintf('case %d: no error (%g)\n', c, r);
  catch ME
    fprintf('case %d: [%s] [%s]\n', c, ME.message, ME.identifier);
  end
end

function r = trigger(c, n)
  v = [1 2 3];
  M = ones(2, 3);
  r = 0;
  for k = 1:n
    switch c
      case 1
        r = r + v(k + 0.5);
      case 2
        r = r + M(k - 3, 1);
      case 3
        r = r + M(1, k + 1);
      case 4
        m = [true false true true];
        w = v(m(1:k + 1));
        r = r + w(1);
      case 5
        u = zeros(1, 4);
        u(1:k) = [1 2];
        r = r + u(1);
      case 6
        P = zeros(2, 2);
        P(1:2, 1:2) = 1:4;
        r = r + P(1);
      case 7
        Q = zeros(3, 3);
        Q(:, k) = ones(1, 3);
        r = r + Q(1);
      case 8
        Q = zeros(3, 3);
        Q(1:2, k) = ones(3, 1);
        r = r + Q(1);
      case 9
        r = r + sum(v, k - 3);
      case 10
        r = r + realsqrt(k - 2);
    end
  end
end
