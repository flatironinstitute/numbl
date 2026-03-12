% Test that [varargout{1:nargout}] = func() pattern works correctly
% This pattern is used extensively in chebfun for delegation to superclass methods
% e.g. [varargout{1:nargout}] = isempty@separableApprox(varargin{:})

% Test 1: delegate returns a scalar logical (nargout=1)
result1 = testDelegate1(5);
assert(result1 == 1, 'delegation of logical scalar should work');

result2 = testDelegate1(3);
assert(result2 == 0, 'delegation of logical scalar false should work');

% Test 2: delegate returns a number (nargout=1)
result3 = testDelegate2(7);
assert(result3 == 14, 'delegation of numeric scalar should work');

% Test 3: delegate with multiple outputs (nargout=3)
[a, b, c] = testDelegate3(10);
assert(a == 10, 'first output should be 10');
assert(b == 20, 'second output should be 20');
assert(c == 30, 'third output should be 30');

disp('SUCCESS');

function varargout = testDelegate1(varargin)
    [varargout{1:nargout}] = innerLogical(varargin{:});
end

function out = innerLogical(x)
    out = (x > 4);
end

function varargout = testDelegate2(varargin)
    [varargout{1:nargout}] = innerNumeric(varargin{:});
end

function out = innerNumeric(x)
    out = x * 2;
end

function varargout = testDelegate3(varargin)
    [varargout{1:nargout}] = innerThree(varargin{:});
end

function [a, b, c] = innerThree(x)
    a = x;
    b = x * 2;
    c = x * 3;
end
