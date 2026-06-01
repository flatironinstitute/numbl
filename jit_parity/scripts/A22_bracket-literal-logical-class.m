% PARITY BUG: multi-element bracket literal of logicals loses its logical class in the JIT.
%
% What it tests: class()/islogical()/isnumeric()/isa()/isfloat() of a
%   multi-element bracket literal whose elements are all logical
%   (`[true, false, true]`, or `[a, b]` where a,b are comparison results).
%
% MATLAB and the numbl interpreter (opt0): such a concatenation is a
%   LOGICAL array -> class 'logical', islogical=1, isnumeric=0.
% numbl JS-JIT (opt1) and C-JIT (opt2): the bracket-literal result is
%   typed DOUBLE -> class 'double', islogical=0, isnumeric=1.
%
% Outputs:
%   opt0: logical
%   opt1: double      <-- DIVERGE
%   opt2: double      <-- DIVERGE
%
% Diverging modes: opt1 (JS-JIT) AND opt2 (C-JIT).
%
% Cause: src/numbl-core/jit/lowering/lowerTensorLit.ts unconditionally
%   builds `tensorDouble(...)` for a multi-element (non-complex) bracket
%   literal (around line 210), ignoring the case where every cell is
%   logical. The interpreter preserves logical when all elements are
%   logical. A single-element literal `[true]` collapses to the scalar
%   and is unaffected; only multi-element concat is wrong. Element-wise
%   results (>, &, |, ~), reductions (any/all), isnan(...) tensor, and
%   true(n,m) all correctly keep logical -- only `[...]` is broken.
%
% JIT engagement: CONFIRMED. `--dump-c` is non-empty (C-JIT engaged) and
%   `%!numbl:assert_jit c` passes at opt2 while still printing 'double'.

function c = f(s)
  v = [true, false, true];
  c = class(v);
end
disp(f(0));
