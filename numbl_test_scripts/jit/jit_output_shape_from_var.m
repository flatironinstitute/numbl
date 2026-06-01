% Regression: a C-JIT `y = x` assign where `x` is a tensor param not
% indexed at 2D+ anywhere in the body used to fall back to
% `[len, 1]` (column) when computing the dest tensor's shape. That
% flipped row vectors to columns and flattened matrices to 1-column
% vectors — the caller saw the wrong `size(y)` even though the data
% contents were right.
%
% The bug lives on the dynamic-output path (`v_y_d0 = v_x_len; v_y_d1 = 1;`
% fallback in emit.ts). It surfaces when the classifier marks `y` as
% `hasFreshAlloc=true` via some Binary/Unary/Call RHS but another assign
% writes `y = x` directly. Each test function seeds y via a fresh-alloc
% then overwrites with a plain alias to hit that path.
%
% Each test function carries a `for` loop so `bodyWorthCrossing` lets the
% hybrid-callees path compile it even when the outer fails feasibility.

%!numbl:assert_jit c
function y = seq_alias(x)
  for i = 1:1
    y = x + 0;   % first assign marks y as fresh/dynamic
    y = x;       % second assign is the Var alias — the bug path
  end
end

% ── Case 1: row vector passed through `y = x` ──────────────────────────
row_in = [1.0 2.0 3.0 4.0 5.0];             % size [1, 5]
for k = 1:20
  r1 = seq_alias(row_in);
end
assert(isequal(size(r1), [1 5]), 'seq_alias row size wrong');
assert(isequal(r1, row_in), 'seq_alias row values wrong');

% ── Case 2: 2D matrix flattened by the column fallback ─────────────────
mat_in = [1.0 2.0 3.0; 4.0 5.0 6.0];        % size [2, 3]
for k = 1:20
  r2 = seq_alias(mat_in);
end
assert(isequal(size(r2), [2 3]), 'seq_alias matrix size wrong');
assert(isequal(r2, mat_in), 'seq_alias matrix values wrong');

% ── Case 3: column vector round-trip (sanity baseline) ─────────────────
col_in = [10.0; 20.0; 30.0; 40.0; 50.0];    % size [5, 1]
for k = 1:20
  r3 = seq_alias(col_in);
end
assert(isequal(size(r3), [5 1]), 'seq_alias column size wrong');
assert(isequal(r3, col_in), 'seq_alias column values wrong');

disp('SUCCESS')
