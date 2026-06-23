% nargin() on handles to user / local / nested functions returns the declared
% input count (negated when the last parameter is varargin), matching how
% nargin resolves a call to the same name.

assert(nargin(@local_two) == 2, 'nargin(@local_two) should be 2');
assert(nargin(@local_none) == 0, 'nargin(@local_none) should be 0');
assert(nargin(@local_varargin) == -2, 'nargin(@local_varargin) should be -2');

h = make_nested();
assert(nargin(h) == 4, 'nargin of nested handle should be 4');

% Anonymous handles keep working.
g = @(x, y, z) x + y + z;
assert(nargin(g) == 3, 'nargin(anon) should be 3');

disp('SUCCESS');

function r = local_two(a, b)
    r = a + b;
end

function r = local_none()
    r = 1;
end

function r = local_varargin(a, varargin)
    r = a;
end

function fn = make_nested()
    fn = @inner;
    function out = inner(w, x, y, z)
        out = w;
    end
end
