% TEST: disp() of a struct with >=2 scalar fields.
%
% opt0 (interp) and opt1 (JS-JIT):
%     a: 5
%     b: 10
%     c: 15
% opt2 (C-JIT):                       <-- DIVERGES (deterministic)
%     a:     b:     c: 5
% 10
% 15
% DIVERGING MODE: opt2 only (opt0==opt1).
%
% Cause: emitNamedTypedef.ts emitStructDisp() prints each field LABEL
% with a raw printf("    <name>: ") (libc stdout) but prints the field
% VALUE via mtoc2_disp_double / disp_tensor, which route through the
% host-write callback (mtoc2_stdout -> mtoc2_host_write). Under a normal
% --opt 2 run the host-write callback is bound (so output is captured),
% so the label channel (raw stdout) and the value channel (host write)
% are flushed independently and the interleaving is lost: all labels
% come out grouped, then all values. A single-field struct happens to
% stay ordered, masking the bug for the simplest shape.
% Fix: route the struct/cell disp labels/punctuation through mtoc2_stdout
% (or never bind the host write while raw printf is in use).
% JIT engagement: CONFIRMED (assert_jit c passes; jsgen=1, cgen=1).
function out = f(n)
  %!numbl:assert_jit c
  s = struct('a', n, 'b', n*2, 'c', n*3);
  disp(s);
  out = n;
end
acc = 0;
for i = 1:1
  acc = acc + f(5);
end
