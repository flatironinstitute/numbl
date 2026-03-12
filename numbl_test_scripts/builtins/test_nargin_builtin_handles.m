% Test nargin() on builtin function handles
assert(nargin(@real) == 1, 'nargin(@real) should be 1');
assert(nargin(@imag) == 1, 'nargin(@imag) should be 1');
assert(nargin(@sin) == 1, 'nargin(@sin) should be 1');
assert(nargin(@cos) == 1, 'nargin(@cos) should be 1');
assert(nargin(@abs) == 1, 'nargin(@abs) should be 1');
assert(nargin(@plus) == 2, 'nargin(@plus) should be 2');
assert(nargin(@minus) == 2, 'nargin(@minus) should be 2');
assert(nargin(@times) == 2, 'nargin(@times) should be 2');
fprintf('SUCCESS\n');
