% TEST: single-element Range slice read `v(2:2)` whose count statically
% folds to 1, assigned to a variable then displayed.
% opt0 (interpreter): prints "20"   (correct)
% opt1 (JS-JIT):      prints "20"   (correct, matches; JS is untyped)
% opt2 (C-JIT):       RuntimeError: mtoc2-c-jit compile failed
%   error: incompatible types when assigning to type 'double' from type
%          'mtoc2_tensor_t'
% DIVERGES: opt2 (errors while opt0/opt1 succeed -> counts as divergence)
% CAUSE: with start/end/step static, mtoc2_loop_count folds to 1, so the
%   type system types the slice result as a scalar 'double' (LHS var y is
%   declared double). But the Range branch of emitIndexSliceProducer
%   (emitIndex.ts ~378-429) ALWAYS allocates and yields a mtoc2_tensor_t,
%   regardless of a statically-scalar result. The C assignment
%   `double y = ({ ... mtoc2_tensor_t _mtoc2_t; ... _mtoc2_t; })` is a type
%   mismatch -> compile failure. (A single-element *Scalar* slot routes to
%   IndexLoad and is fine; only a count-folds-to-1 Range slot hits this.)
%   Also reproduces with disp(v(2:2)), v(1:1), folded lo=2;hi=2; v(lo:hi),
%   and matrix A(2:2,1).
% JIT-ENGAGEMENT: confirmed - opt2 reaches the C compiler; the failure is
%   in the generated C (dump-c non-empty, ~628 lines).
v=[10 20 30];
y=v(2:2);
disp(y);
