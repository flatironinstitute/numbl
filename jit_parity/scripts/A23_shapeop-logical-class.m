% AREA: shape/structural builtins — element-class (logical) preservation
%
% WHAT IT TESTS: class() of a logical matrix after a transpose. The
% interpreter keeps the result `logical`; the JIT shape-builtin transfer
% converts it to `double`, then folds class(...) to the literal "double".
%
% OUTPUTS:
%   opt0 (interp, REFERENCE): logical
%   opt1 (JS-JIT):            double      <-- DIVERGES
%   opt2 (C-JIT):             double      <-- DIVERGES
%
% Same divergence reproduces for reshape, repmat, cat(dim,..) and vertcat
% `[L;L]` (all preserve logical in opt0, become double in the JIT). flip,
% diag, tril and sort already drop logical in opt0 too, so those AGREE.
%
% HYPOTHESIZED CAUSE: transpose/reshape/repmat/cat transfer() in
% src/numbl-core/jit/builtins/defs/shape/*.ts always return tensorDouble*
% (elem "double"), discarding the input's `logical` elem. The interpreter
% (array-manipulation.ts) preserves logical for these ops. Note: when the
% true class IS logical the JIT can't emit "logical" from class(), so it
% DECLINES (falls back) — only the wrongly-double cases engage and diverge.
%
% JIT ENGAGEMENT: CONFIRMED. opt1 --dump-js is 95 lines with
% mtoc2_tensor_transpose(L) and a folded {value:"double"}; opt2 --dump-c is
% 715 lines; `%!numbl:assert_jit c` passes (C-JIT engages).
L = [true false; true true];
disp(class(L'));
