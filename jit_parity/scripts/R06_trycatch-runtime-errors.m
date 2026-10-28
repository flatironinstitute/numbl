% R: common runtime errors raised inside JIT'd code and caught, printing
% ME.message and ME.identifier. Each case runs in a JIT'd function (call
% shape: the error is raised directly) and in a loop inside the try (loop
% shape: the unit re-runs on the interpreter, whose state the catch block
% can observe).
for c = 1:16
  try
    r = trigger(c, 4);
    fprintf('case %d: no error (%g)\n', c, r);
  catch ME
    fprintf('case %d: [%s] [%s]\n', c, ME.message, ME.identifier);
  end
end

v = [1 2 3];
s = 0;
try
  for k = 1:5
    s = s + v(k);
  end
catch ME
  fprintf('loop: [%s] [%s] s=%g k=%d\n', ME.message, ME.identifier, s, k);
end
try
  for k = 1:5
    assert(k < 3, 'parity:assert', 'k=%d failed', k);
  end
catch ME
  fprintf('loop assert: [%s] [%s] k=%d\n', ME.message, ME.identifier, k);
end

function r = trigger(c, n)
  v = [1 2 3];
  M = ones(2, 3);
  r = 0;
  for k = 1:n
    switch c
      case 1
        r = r + v(k);
      case 2
        r = r + M(k, 1);
      case 3
        r = r + M(1, k);
      case 4
        r = r + v(k - 1);
      case 5
        w = ones(1, k) + v;
        r = r + w(1);
      case 6
        P = ones(2, k) * ones(3, 2);
        r = r + P(1);
      case 7
        Q = [ones(2, 2), ones(k, 1)];
        r = r + Q(1);
      case 8
        R = reshape(1:6, k, 2);
        r = r + R(1);
      case 9
        if k == 3
          error('parity:id', 'fmt %d of %d', k, n);
        end
      case 10
        if k == 3
          error('literal %d message');
        end
      case 11
        assert(k < 3);
      case 12
        assert(k < 3, 'plain message');
      case 13
        assert(k < 3, 'parity:a', 'value %d', k);
      case 14
        Z = zeros(1, k / 2);
        r = r + numel(Z);
      case 15
        u = zeros(1, 3);
        u(1:k) = ones(1, 3);
        r = r + u(1);
      case 16
        r = r + dot(v, ones(1, k));
    end
  end
end
