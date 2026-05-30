% DIAGNOSIS: C-JIT `deal` (jit/builtins/defs/system/deal.ts emitC) writes each
% output through its pointer in source order, joined by the comma operator,
% with NO temporaries:  ((void)((*&a = b), (*&b = a))). When a later source
% expression reads a variable an earlier write already overwrote, it sees the
% wrong value -- so the standard swap idiom [a,b]=deal(b,a) corrupts. The JS
% path (emitJs) builds an array literal first and the interpreter snapshots, so
% only opt2 diverges. (Top-level C-JIT engages only when every statement is
% suppressed, hence the trailing ';' on the fprintf.)
%
% --opt 0/1 output (correct):  "2 1"
% --opt 2 output (buggy):      "2 2"
a = 1;
b = 2;
[a, b] = deal(b, a);
fprintf('%d %d\n', a, b);
