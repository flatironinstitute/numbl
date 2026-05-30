% repelem(v, counts) with a per-element count vector must repeat each element
% the corresponding number of times. numbl only implemented the scalar-count
% form (Math.round(toNumber(...))), crashing on a vector of counts.

assert(isequal(repelem([1 2 3], [3 1 2]), [1 1 1 2 3 3]), 'repelem per-element counts');
assert(isequal(repelem([10 20], [2 3]), [10 10 20 20 20]), 'repelem counts 2');

% a zero count drops that element
assert(isequal(repelem([1 2 3], [2 0 1]), [1 1 3]), 'repelem zero count');

% scalar count still works (count applies to every element)
assert(isequal(repelem([1 2 3], 2), [1 1 2 2 3 3]), 'repelem scalar count');

% column vector input
assert(isequal(repelem([1; 2], [2; 1]), [1; 1; 2]), 'repelem column vector');

disp('SUCCESS');
