% C-JIT parity gap #23: complex fused chain had a read-after-write bug
% on reassignment. For `y = y .* x` (second assign to `y`), the fused
% codegen emitted
%   __f_y_re = RHS_re(__f_y_re, __f_y_im, ...);
%   __f_y_im = RHS_im(__f_y_re, __f_y_im, ...);
% in that order, so the second line read the *already updated* __f_y_re
% instead of the old one, corrupting the imaginary part.
%
% Expected disp output (must match across all runs):
%   numbl --opt 1 run <this>             -> SUCCESS
%   numbl --opt 2 run <this>             -> SUCCESS
%   numbl --opt 2 --fuse run <this>      -> SUCCESS
%   matlab -batch parity23_complex_fused_aliasing -> SUCCESS

x = [1+1i, 2+2i, 3+3i];

% chain(x) = (x+2) .* x
% For x=1+1i: (3+1i)(1+1i) = 2+4i
% For x=2+2i: (4+2i)(2+2i) = 4+12i
% For x=3+3i: (5+3i)(3+3i) = 6+24i
for k = 1:4
    a = chain(x);
end
assert(isequal(a, [2+4i, 4+12i, 6+24i]), 'complex fused reassign');

% y = x.*x; y = y + x  (pure-real update — cross-check)
for k = 1:4
    b = chain2(x);
end
assert(isequal(b, [1+3i, 2+10i, 3+21i]), 'complex fused add-reassign');

% Three-step chain exercises alias through multiple reassignments.
for k = 1:4
    c = chain3(x);
end
assert(isequal(c, [2+4i, 4+12i, 6+24i]), 'complex fused 3-step');

disp('SUCCESS')

function y = chain(x)
    y = x + 2;
    y = y .* x;
end

function y = chain2(x)
    y = x .* x;
    y = y + x;
end

function y = chain3(x)
    y = x + 1;
    y = y .* x;
    y = y + x;
end
