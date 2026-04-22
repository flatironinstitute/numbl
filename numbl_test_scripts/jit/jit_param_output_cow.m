% Regression: the JIT must not mutate the caller's tensor when a function
% with `function a = f(a)` (param name == output name) is called from a
% JIT-hot loop. The JS-JIT unshare(t) path returned the original tensor
% when _rc == 1, and callUser never bumped _rc on its args, so writes
% inside the callee reached the caller's buffer.

function a = scalar_index_assign(a)
  for k = 1:length(a)
    a(k) = a(k) * 2.0;
  end
end

function a = two_d_index_assign(a)
  a(2, 2) = 99.0;
  a(1, 3) = 77.0;
end

function a = col_assign(a, col)
  a(:, 2) = col;
end

function a = range_assign(a)
  a(2:4) = a(5:7);
end

function a = whole_tensor_expr(a, b)
  a = a .* b + 1.0;
end

% ── Case 1: scalar AssignIndex inside a for-loop (1D param-output) ─────
x1 = [1.0; 2.0; 3.0; 4.0];
x1_copy = x1;
for k = 1:20
  y1 = scalar_index_assign(x1);
end
assert(isequal(x1, x1_copy), 'scalar_index_assign must not mutate x');
assert(isequal(y1, [2.0; 4.0; 6.0; 8.0]), 'scalar_index_assign result wrong');

% ── Case 2: 2D AssignIndex ─────────────────────────────────────────────
x2 = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0];
x2_copy = x2;
for k = 1:20
  y2 = two_d_index_assign(x2);
end
assert(isequal(x2, x2_copy), 'two_d_index_assign must not mutate x');
assert(y2(2, 2) == 99.0 && y2(1, 3) == 77.0, 'two_d_index_assign result wrong');
assert(x2(2, 2) == 5.0 && x2(1, 3) == 3.0, 'x must still hold original values');

% ── Case 3: AssignIndexCol (dst(:, j) = src) ───────────────────────────
x3 = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0];
x3_copy = x3;
c3 = [100.0; 200.0; 300.0];
for k = 1:20
  y3 = col_assign(x3, c3);
end
assert(isequal(x3, x3_copy), 'col_assign must not mutate x');
assert(isequal(y3(:, 2), c3), 'col_assign result wrong');

% ── Case 4: AssignIndexRange (dst(a:b) = src(c:d), overlapping) ───────
x4 = [1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0];
x4_copy = x4;
for k = 1:20
  y4 = range_assign(x4);
end
assert(isequal(x4, x4_copy), 'range_assign must not mutate x');
assert(isequal(y4, [1.0; 5.0; 6.0; 7.0; 5.0; 6.0; 7.0; 8.0]), ...
  'range_assign result wrong');

% ── Case 5: tensor-valued Assign (`a = a .* b + 1`) ────────────────────
x5 = [1.0; 2.0; 3.0; 4.0];
x5_copy = x5;
b5 = [10.0; 20.0; 30.0; 40.0];
for k = 1:20
  y5 = whole_tensor_expr(x5, b5);
end
assert(isequal(x5, x5_copy), 'whole_tensor_expr must not mutate x');
assert(isequal(y5, [11.0; 41.0; 91.0; 161.0]), 'whole_tensor_expr result wrong');

disp('SUCCESS')
