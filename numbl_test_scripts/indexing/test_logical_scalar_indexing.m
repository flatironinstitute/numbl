% Test indexing into a logical scalar value.
% In MATLAB, a scalar logical is treated as a 1x1 array and supports:
%   x(1)     → x itself
%   x(:)     → x itself
%   x([1,1]) → logical array with x.value repeated

t = true;
f = false;

% Basic scalar index
r1 = t(1);
assert(r1 == true, 'true(1) should be true');

r2 = f(1);
assert(r2 == false, 'false(1) should be false');

% Colon index
r3 = t(:);
assert(r3 == true, 'true(:) should be true');

r4 = f(:);
assert(r4 == false, 'false(:) should be false');

% Tensor index (all indices must be 1 for a scalar)
r5 = t([1, 1, 1]);
assert(isequal(r5, logical([1, 1, 1])), 'true([1,1,1]) should be [true,true,true]');

r6 = f([1, 1]);
assert(isequal(r6, logical([0, 0])), 'false([1,1]) should be [false,false]');

% Logical scalar index
r7 = t(true);
assert(r7 == true, 'true(true) should be true');

% Empty tensor index → empty result
r8 = t(zeros(1,0));
assert(isempty(r8), 'true([]) should be empty');

disp('SUCCESS');
