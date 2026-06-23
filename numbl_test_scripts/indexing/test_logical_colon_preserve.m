% Reshaping a logical array with the colon index, x(:), must preserve the
% logical class.  Otherwise x(:) becomes a double of 0/1 values and using it
% as a mask, m(x(:)) = ..., reads it as numeric indices (0 => out of bounds).
%
% This is the mesh2d/smooth2.m pattern:
%   free = (vdeg == 0);     % logical
%   keep(free(:)) = true;   % logical-mask assignment
% which threw "Index exceeds array bounds" when free(:) lost its logical class.

vdeg = [0; 2; 0; 3];
free = (vdeg == 0);

% --- colon preserves logical class ---
assert(islogical(free), 'free should be logical');
fc = free(:);
assert(islogical(fc), 'free(:) should stay logical');
assert(isequal(size(fc), [4 1]), 'free(:) shape');

% --- logical-mask assignment via colon (the smooth2 use) ---
keep = false(4, 1);
keep(free(:)) = true;
assert(islogical(keep), 'keep stays logical');
assert(isequal(keep, logical([1; 0; 1; 0])), 'mask assignment via free(:)');

% --- logical-mask read via colon ---
data = [10; 20; 30; 40];
sel = data(free(:));
assert(isequal(sel, [10; 30]), 'masked read via free(:)');

% --- colon on a 2-D logical, flattens column-major and stays logical ---
L = logical([1 0; 1 1]);
Lc = L(:);
assert(islogical(Lc), 'L(:) logical');
assert(isequal(Lc, logical([1; 1; 0; 1])), 'L(:) column-major values');

% --- a plain numeric array must NOT become logical through colon ---
n = [1 2 3];
nc = n(:);
assert(~islogical(nc), 'numeric stays numeric through colon');

disp('SUCCESS');
