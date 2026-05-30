% TEST: a SCALAR logical used as an index — `A(false)` / `A(true)` (read)
% and `A(true) = x` (write). A scalar logical is a 1-element logical mask
% in MATLAB (`A(false)` -> empty, `A(true)` -> A(1)), not a positional
% index.
% opt0 (interp): prints "0 99 20"   (correct: y empty, A(1) set to 99)
% opt1 (JS-JIT): prints "0 99 20"   (correct — but only via a RUNTIME bail:
%                  the emitted JS computes index round(false)=0, throws a
%                  RangeError, and the scope falls back to the interpreter)
% opt2 (C-JIT):  RuntimeError: Index exceeds array bounds (got 0, valid 1..5)
%                  — the C kernel aborts; no fallback.   <-- DIVERGES
% DIVERGING MODE: opt2 (hard error while opt0/opt1 produce output).
% CAUSE: isScalarRealNumeric() accepts a scalar logical, so lowerIndexLoad/
%   lowerIndexStore/lowerIndexSlice classified `A(false)` as a POSITIONAL
%   read of index 0 (false -> 0.0) instead of a logical mask. opt1 escaped
%   via a runtime bail; opt2's mtoc2_idx_lin aborts. The logical-mask
%   codegen needs a tensor and scalar logicals are bare doubles, so the fix
%   DECLINES the scalar-logical index to the interpreter (a clean
%   compile-time decline, like the other unsupported index forms). The
%   multi-element logical-mask path is unchanged.
% JIT-ENGAGEMENT: the construct now declines to the interpreter (by design);
%   the point of this script is that no mode crashes.
function test()
  A = [10 20 30 40 50];
  c = false;
  y = A(c);
  A(true) = 99;
  fprintf('%d %d %d\n', numel(y), A(1), A(2));
end
test();
