% TEST: bracket/cat of a scalar with a compile-time empty: C = [5, []].
% MATLAB drops the empty -> C is the scalar 5 (1x1).
% opt0 (interp): 1x1 val=5
% opt1 (JS-JIT): 1x1 val=5
% opt2 (C-JIT):  RuntimeError "mtoc2-c-jit compile failed" with gcc errors
%   "request for member 'dims' in something not a structure or union"
%   on C.dims[0]  <-- DIVERGES (compile failure ESCAPES the JS fallback)
% DIVERGING MODE: opt2 only.
%
% Cause: when [scalar, []] drops the empty, the result types as a scalar
%   double, but size(C,1)/size(C,2) codegen still emits C.dims[i] member
%   access (tensor form) on the scalar var -> uncompilable C. The
%   scalar-result-vs-tensor-codegen mismatch isn't reconciled, and the
%   compile error is surfaced to the user instead of falling back to JS.
%   Also reproduces with cat(2,5,[]) and cat(3,5,[]).
% JIT engagement: CONFIRMED (reaches the C compiler, which then fails).
C = [5, []];
fprintf('%dx%d val=%g\n', size(C,1), size(C,2), C);
