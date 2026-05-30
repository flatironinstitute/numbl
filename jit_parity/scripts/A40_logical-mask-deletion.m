% TEST: element deletion via a logical-mask variable: x(m) = [].
% MATLAB: removes the masked elements -> x = [2 4].
% opt0 (interp): 2 4   (correct)
% opt1 (JS-JIT): NaN 2 NaN 4 NaN   <-- DIVERGES (masked store from a
%               0-element RHS -> NaN garbage)
% opt2 (C-JIT):  error "Subscripted assignment dimension mismatch"
% DIVERGING MODE: all three differ.
%
% Cause: the JIT has no `= []` deletion detection (deletion is implemented
%   only in the interpreter). lowerIndexSliceStore compiles `x(m) = []` as
%   an ordinary masked store against a 0-element tensor: JS reads NaN from
%   the empty RHS, C emits the tensor-RHS size check and aborts.
% FIX DIRECTION: the JIT should DECLINE (UnsupportedConstruct) any indexed
%   store whose RHS is an empty `[]` literal, so all three run the
%   interpreter's correct deletion.
% JIT engagement: CONFIRMED (C-JIT engaged; ~895-line dump).
x = [1 2 3 4 5];
m = logical([1 0 1 0 1]);
x(m) = [];
disp(x);
