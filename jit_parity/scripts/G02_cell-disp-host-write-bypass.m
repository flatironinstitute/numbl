% TEST: disp() of a cell array interleaved with fprintf output.
%
% opt0 (interp) and opt1 (JS-JIT):
%   before 5
%   {5, 10}
%   after 5
% opt2 (C-JIT):              <-- DIVERGES (deterministic)
%   {5, 10}
%   before 5
%   after 5
% DIVERGING MODE: opt2 only (opt0==opt1).
%
% Cause: the generated cell _disp helper (emitCellTypedef.ts) prints the
% braces/commas/scalar slots with raw printf()/fputs(...,stdout) and the
% tensor slots via mtoc2_disp_tensor_inline (also raw stdout). All of
% that goes to libc stdout, while fprintf and disp(scalar) route through
% the host-write callback (mtoc2_stdout -> mtoc2_host_write). Under a
% captured --opt 2 run the two channels are flushed independently, so the
% cell's whole output jumps ahead of the surrounding fprintf/disp lines.
% Same root cause as 04 (raw printf bypasses the host-write hook).
% JIT engagement: CONFIRMED (assert_jit c passes; jsgen=1, cgen=1).
function out = f(n)
  %!numbl:assert_jit c
  fprintf('before %d\n', n);
  c = {n, n*2};
  disp(c);
  fprintf('after %d\n', n);
  out = n;
end
acc = 0;
for i = 1:1
  acc = acc + f(5);
end
