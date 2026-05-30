% Regression: C-JIT --fuse must preserve the prelude's unshare copy of a
% param-output tensor so the fused loop body can read the input values.
% The real-fused-chain emitter used to unconditionally `free(v_y_data)` +
% `malloc(...)` before the loop, clobbering the memcpy'd input and leaving
% uninitialized memory for the fused expression to read.
%
% The outer for-loop is what triggers the C-JIT hybrid compile; without
% it, this function is only called once and JS-JIT handles it. Twenty
% iterations is enough to tip over the compile threshold.

%!numbl:assert_jit c
function y = fuse_self(y, x)
  y = y .* x + 3.0;
end

function y = fuse_two_ops(y, x)
  y = y + x + 1.0;
end

function [y, z] = fuse_two_outs(y, x)
  z = y + x;
  y = z + 1.0;
end

function y = fuse_self_scalar_only(y)
  y = y + 1.0;
  y = y * 2.0;
end

% ── Case 1: y = y .* x + 3 ─────────────────────────────────────────────
a1 = [1.0; 2.0; 3.0; 4.0];
b1 = [10.0; 20.0; 30.0; 40.0];
a1_copy = a1;
for k = 1:20
  r1 = fuse_self(a1, b1);
end
assert(isequal(a1, a1_copy), 'fuse_self must not mutate the caller arg');
assert(isequal(r1, [13.0; 43.0; 93.0; 163.0]), 'fuse_self result wrong');

% ── Case 2: y = y + x + 1 (the minimal repro) ──────────────────────────
a2 = [100.0; 200.0; 300.0];
b2 = [1.0; 2.0; 3.0];
a2_copy = a2;
for k = 1:20
  r2 = fuse_two_ops(a2, b2);
end
assert(isequal(a2, a2_copy), 'fuse_two_ops must not mutate the caller arg');
assert(isequal(r2, [102.0; 203.0; 304.0]), 'fuse_two_ops result wrong');

% ── Case 3: z = y + x; y = z + 1 (two paramOutputs, same chain) ────────
a3 = [10.0; 20.0; 30.0];
b3 = [1.0; 2.0; 3.0];
a3_copy = a3;
for k = 1:20
  [u, v] = fuse_two_outs(a3, b3);
end
assert(isequal(a3, a3_copy), 'fuse_two_outs must not mutate the caller arg');
assert(isequal(v, [11.0; 22.0; 33.0]), 'fuse_two_outs v wrong');
assert(isequal(u, [12.0; 23.0; 34.0]), 'fuse_two_outs u wrong');

% ── Case 4: two separate fused chains in one function ─────────────────
a4 = [10.0; 20.0; 30.0];
a4_copy = a4;
for k = 1:20
  r4 = fuse_self_scalar_only(a4);
end
assert(isequal(a4, a4_copy), 'fuse_self_scalar_only must not mutate the caller arg');
assert(isequal(r4, [22.0; 42.0; 62.0]), 'fuse_self_scalar_only result wrong');

disp('SUCCESS')
