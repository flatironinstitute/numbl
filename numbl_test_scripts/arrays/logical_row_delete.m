% Test 2D logical row deletion: r(mask, :) = []
% In MATLAB, r(logical_mask, :) = [] removes rows where the mask is true.

% Column vector case
r = [1; 2; 3; 4; 5];
mask = logical([0; 0; 1; 0; 1]);
r(mask, :) = [];
assert(isequal(r, [1; 2; 4]), 'should remove rows 3 and 5 from column vector');

% Matrix case
M = [10 11; 20 21; 30 31; 40 41];
mask2 = logical([1; 0; 1; 0]);
M(mask2, :) = [];
assert(isequal(M, [20 21; 40 41]), 'should remove rows 1 and 3 from matrix');

% NaN row removal pattern (used by chebfun roots)
r2 = [-0.8; -0.5; -0.3; 0.04; 0.2; 0.6; 0.7; NaN; NaN];
r2(all(isnan(r2), 2), :) = [];
assert(length(r2) == 7, 'should have 7 elements after NaN removal');
assert(~any(isnan(r2)), 'should have no NaN after removal');
assert(r2(1) == -0.8, 'first element should be -0.8');

% Remove none
w = [1; 2; 3];
w(logical([0; 0; 0]), :) = [];
assert(isequal(w, [1; 2; 3]), 'removing none should keep all');

% Remove all
z = [1; 2; 3];
z(logical([1; 1; 1]), :) = [];
assert(isempty(z), 'removing all should give empty');

disp('SUCCESS');
