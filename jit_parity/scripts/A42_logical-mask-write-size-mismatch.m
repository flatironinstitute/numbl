% TEST: a logical-mask STORE with a tensor RHS whose length != the number
% of truthy mask bits: a(logical([1 1 1 0 0 0])) = [7 8]  (3 slots, 2 vals).
% MATLAB / opt0 (interp): error "Subscripted assignment dimension mismatch".
% opt1 (JS-JIT, before fix): SILENTLY wrote [7 8 NaN 4 5 6]   <-- DIVERGED
% opt2 (C-JIT): already errored via emitTensorRhsSizeCheck.
% DIVERGING MODE: opt1 only (silent NaN corruption / truncation).
%
% Cause: emitJs.ts single-slot LogicalMask store omitted the
%   numel(rhs)==nnz(mask) check that the Range store (and the C path)
%   already had. FIX: add the length check so the JS-JIT mask store errors
%   like opt0/opt2.
% JIT engagement: the for-loop routes the mask store through jit-loop, so
%   pre-fix opt1 reached the unchecked codegen (confirmed: it wrote the NaN
%   array). All three modes now agree (error -> <ERROR>).
a = [1 2 3 4 5 6];
for k = 1:1
    a(logical([1 1 1 0 0 0])) = [7 8];
end
disp(a);
