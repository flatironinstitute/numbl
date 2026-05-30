% TEST: row/column/colon deletion: A(:,2) = [] (delete a column).
% MATLAB: A becomes [1 3; 4 6; 7 9].
% opt0 (interp): 1 3 / 4 6 / 7 9   (correct)
% opt1 (JS-JIT): correct (matches opt0)
% opt2 (C-JIT):  error "Subscripted assignment dimension mismatch"  <-- DIVERGES
% DIVERGING MODE: opt2 only. (x(:)=[] full-deletion is a sibling: there
%   opt1 no-ops while opt0 empties; x(2:3)=[] range deletion errors at opt2.)
%
% Cause: same missing `= []` deletion detection as A40 -- the multi-slot /
%   colon / range store path hits the C tensor-RHS size check against the
%   0-element [].
% FIX DIRECTION: DECLINE any `<indexed> = []` store to the interpreter.
% JIT engagement: CONFIRMED (C-JIT engaged; ~769-line dump).
A = [1 2 3; 4 5 6; 7 8 9];
A(:,2) = [];
disp(A);
