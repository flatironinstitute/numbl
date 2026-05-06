% Regression: a C-JIT fixed tensor output whose length differs from the
% first tensor param's length triggers a heap overflow / truncated return.
% The JS wrapper in cJitInstall.ts used to size the fixed-output buffer
% to `allocFloat64Array(firstTensorLen)`, so `y = x2` with len(x2) != len(x1)
% would memcpy the wrong number of doubles into a mis-sized Float64Array.
%
% Each test function contains a `for` loop so `bodyWorthCrossing` lets the
% hybrid-callees path compile it even when the outer fails feasibility;
% otherwise trivial `y = x2` callees skip C-JIT entirely.
%
% The outer for-loop tips the compile into C-JIT hybrid mode.

function y = return_second_loopy(x1, x2)
  % Loop makes the body substantial enough for bodyWorthCrossing to accept
  % this function for standalone C-JIT compilation. The assign `y = x2`
  % is what actually exercises the output-buffer sizing path.
  for i = 1:1
    y = x2;
  end
end

function y = return_first_loopy(x1, x2)
  for i = 1:1
    y = x1;
  end
end

function [a, b] = return_both(x1, x2)
  a = x1;
  b = x2;
end

function y = paramout_replace(y, x)
  % y is paramOutput, but `y = x` makes it a Var alias (hasFreshAlloc=false
  % in classify), so it falls off the dynamic-output path even though the
  % JS wrapper seeds the buffer from y's own input length (not x's).
  for i = 1:1
    y = x;
  end
end

% ── Case 1: output length > firstTensorLen (would heap-overflow) ───────
a1 = [1.0 2.0 3.0];                     % firstTensorLen = 3
b1 = [10.0 20.0 30.0 40.0 50.0];        % actual output length = 5
for k = 1:20
  r1 = return_second_loopy(a1, b1);
end
assert(isequal(r1, b1), 'return_second_loopy must return x2 intact (len > firstTensorLen)');

% ── Case 2: output length < firstTensorLen (would return stale/uninit) ─
a2 = [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0]; % firstTensorLen = 8
b2 = [99.0 100.0];                       % actual output length = 2
for k = 1:20
  r2 = return_second_loopy(a2, b2);
end
assert(isequal(r2, b2), 'return_second_loopy must return x2 intact (len < firstTensorLen)');

% ── Case 3: first-param return is OK (sanity baseline) ─────────────────
a3 = [1.0 2.0 3.0];
b3 = [10.0 20.0 30.0 40.0 50.0];
for k = 1:20
  r3 = return_first_loopy(a3, b3);
end
assert(isequal(r3, a3), 'return_first_loopy must return x1 intact');

% ── Case 4: multi-output with mismatched lengths ───────────────────────
a4 = [1.0 2.0 3.0];
b4 = [10.0 20.0 30.0 40.0 50.0];
for k = 1:20
  [u4, v4] = return_both(a4, b4);
end
assert(isequal(u4, a4), 'return_both first output must match a4');
assert(isequal(v4, b4), 'return_both second output must match b4');

% ── Case 5: paramOutput `y = x` where x longer than y's own input ──────
y5 = [1.0 2.0 3.0];                      % paramOutput seed len = 3
x5 = [10.0 20.0 30.0 40.0 50.0];         % actual output length = 5
for k = 1:20
  r5 = paramout_replace(y5, x5);
end
assert(isequal(r5, x5), 'paramout_replace must return x intact');

disp('SUCCESS')
