% TEST: logical-mask READ where the mask is LONGER than the base and has a
% truthy bit past the end: x = [1 2 3]; m = logical([1 0 1 0 1]); x(m).
% MATLAB: ERRORS ("The logical indices contain a true value outside of the
%   array bounds").
% opt0 (interp): 1 3 NaN   <-- WRONG (reads past the buffer as NaN, no error)
% opt1 (JS-JIT): 1 3 NaN   <-- WRONG (same)
% opt2 (C-JIT):  error "Index exceeds array bounds"  (correct, matches MATLAB)
% DIVERGING MODE: opt0 and opt1 (they silently read OOB; opt2 is right).
%
% Cause: the interpreter / JS-JIT logical-mask read iterates the mask by
%   its own length and reads positions past numel(base) (yielding NaN)
%   instead of bounds-checking each truthy bit. C's
%   mtoc2_logical_mask_indices correctly bounds-checks.
% FIX DIRECTION: make the interpreter + JS-JIT logical-mask read error on a
%   truthy bit at index > numel(base), matching opt2/MATLAB. (Trailing
%   FALSE bits past the end stay fine in all modes.)
x = [1 2 3];
m = logical([1 0 1 0 1]);
disp(x(m));
