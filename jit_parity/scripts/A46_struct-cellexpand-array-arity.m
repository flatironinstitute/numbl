% TEST: the struct(name, {a,b,c}) cell-expansion form, which in MATLAB
% builds a 1x3 STRUCT ARRAY (one element per cell entry): struct('a',{1,2,3}).
% opt0 (interp): a proper 1x3 struct array (reference behavior == MATLAB).
% opt1 (JS-JIT): leaks raw cell internals on disp   <-- DIVERGES
% opt2 (C-JIT):  "a: {1, 2, 3}" (a SCALAR struct holding a cell)  <-- DIVERGES
% DIVERGING MODE: opt1 and opt2 (both mis-type it as a scalar struct).
%
% Cause: the JIT lowering of the struct(name,{...}) cell-expansion form does
%   not produce a struct array; it stores the cell as the scalar field
%   value. (Contained to display today -- numel(s)/s(2).a force interp
%   fallback -- but the arity is wrong.)
% FIX DIRECTION: DECLINE the struct(...) cell-expansion (struct-array)
%   construction form to the interpreter.
% (wrapped in a 1-iter loop so the scope JIT-compiles.)
for i = 1:1
  s = struct('a', {1, 2, 3});
  disp(s);
end
