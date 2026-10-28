% nargin / nargout / narginchk outside a function raise MATLAB's error;
% inside an anonymous function they are the anonymous function's own.
try
    n = nargin; %#ok<NASGU>
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:narginout:NoStack'));
    assert(strcmp(e.message, 'You can only call nargin/nargout from within a MATLAB function.'));
end
try
    n = nargout; %#ok<NASGU>
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:narginout:NoStack'));
end
try
    narginchk(0, 1);
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:narginout:NoStack'));
end
try
    eval('nargin');
    error('expected an error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:narginout:NoStack'));
end

f = @() nargin;
assert(f() == 0);
g = @(a, b) nargin;
assert(g(1, 2) == 2);
assert(g(1) == 1);
assert(two_args(1, 2) == 0);
assert(count_args(1, 2, 3) == 3);
assert(count_args() == 0);
disp('SUCCESS');

function r = two_args(a, b) %#ok<INUSD>
    h = @() nargin;
    r = h();
end

function n = count_args(varargin)
    n = nargin;
end
