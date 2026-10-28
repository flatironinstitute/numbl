% arguments blocks: positional argument validation. Class conversion, then
% size (with scalar expansion and vector reshaping), then validators, with
% MATLAB's error messages and identifiers. Verified against MATLAB R2025b.

% --- validators ---
assert(fpos(3) == 3);
check_err(@() fpos(-3), 'MATLAB:validators:mustBePositive', ...
  'Invalid argument at position 1. Value must be positive.');
check_err(@() fmulti(1, -2), 'MATLAB:validators:mustBePositive', ...
  'Invalid argument at position 2. Value must be positive.');
check_err(@() frange(11), 'MATLAB:validators:mustBeInRange', ...
  'Invalid argument at position 1. Value must be greater than or equal to 0, and less than or equal to 10.');
check_err(@() fmember('c'), 'MATLAB:validators:mustBeMember', ...
  sprintf('Invalid argument at position 1. Value must be a member of this set:\n    ''a''\n    ''b''\n'));
assert(strcmp(fmember('a'), 'a'));
check_err(@() fnumeric('a'), 'MATLAB:validators:mustBeNumeric', ...
  'Invalid argument at position 1. Value must be numeric.');
% a validator may reference an earlier argument
assert(fless(5, 3) == 3);
check_err(@() fless(1, 3), 'MATLAB:validators:mustBeLessThan', ...
  'Invalid argument at position 2. Value must be less than 1.');
% user-defined validators keep their identifier; the message is prefixed
check_err(@() fuserval(1), 'my:id', ...
  'Invalid argument at position 1. custom message 1');
check_err(@() fuserval2(1), '', 'Invalid argument at position 1. Value too small');
assert(fuserval2(7) == 7);

% --- class conversion ---
x = fpos(true);
assert(isa(x, 'double') && x == 1);
x = fpos("5");
assert(isa(x, 'double') && x == 5);
check_err(@() fpos("abc"), 'MATLAB:validators:mustBePositive', ...
  'Invalid argument at position 1. Value must be positive.');
check_err(@() fpos({1}), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type double or be convertible to double.');
x = fdbl('ab');
assert(isequal(x, [97 98]) && isa(x, 'double'));
check_err(@() fdbl(struct('a', 1)), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type double or be convertible to double.');
check_err(@() fdbl(@sin), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type double or be convertible to double.');
x = fchar("hi");
assert(ischar(x) && strcmp(x, 'hi'));
x = fchar({'a'});
assert(ischar(x) && strcmp(x, 'a'));
check_err(@() fchar({1}), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type char or be convertible to char.');
x = fstr('hi');
assert(isstring(x) && x == "hi");
x = fstr(5);
assert(isstring(x) && x == "5");
x = fstr({'a', 'b'});
assert(isstring(x) && isequal(x, ["a", "b"]));
check_err(@() fstr(struct()), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type string or be convertible to string.');
x = flog(2);
assert(islogical(x) && x);
x = flog(0);
assert(islogical(x) && ~x);
x = flog([1 0 3]);
assert(islogical(x) && isequal(x, [true false true]));
check_err(@() flog(NaN), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type logical or be convertible to logical.');
check_err(@() flog("true"), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type logical or be convertible to logical.');
check_err(@() fcell(1), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type cell or be convertible to cell.');
assert(iscell(fcell({1})));
check_err(@() fstruct(1), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type struct or be convertible to struct.');
check_err(@() ffh(1), 'MATLAB:validation:UnableToConvert', ...
  'Invalid argument at position 1. Value must be of type function_handle or be convertible to function_handle.');
% class conversion happens before the size check
check_err(@() fc11("ab"), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a scalar.');
assert(isequal(fs12({'a', 'b'}), ["a", "b"]));

% --- size ---
check_err(@() fpos([1 2]), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a scalar.');
check_err(@() fpos([]), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a scalar.');
% vectors are reshaped to the declared orientation
assert(isequal(frow([1; 2; 3]), [1 2 3]));
assert(isequal(fcol([1 2 3]), [1; 2; 3]));
assert(isequal(size(frow([])), [1 0]));
assert(isequal(size(frow(zeros(0, 1))), [1 0]));
assert(isequal(size(fcol([])), [0 1]));
assert(isequal(frow(ones(1, 1, 3)), [1 1 1]));
assert(isequal(f13([1; 2; 3]), [1 2 3]));
assert(isequal(f31([1 2 3]), [1; 2; 3]));
r = frow({1; 2});
assert(iscell(r) && isequal(size(r), [1 2]));
check_err(@() frow(ones(2, 2)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a vector.');
check_err(@() fcol(ones(2, 2)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a vector.');
check_err(@() f13([1 2]), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a vector with 3 elements.');
check_err(@() f31([1 2]), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a vector with 3 elements.');
% scalars expand to a fully specified size (':' taken as 1)
assert(isequal(f33(5), 5 * ones(3)));
assert(isequal(f13(5), [5 5 5]));
assert(isequal(fx3(5), [5 5 5]));
assert(isequal(f23c(5), 5 * ones(2, 3)));
assert(isequal(size(f234(7)), [2 3 4]));
check_err(@() f33(ones(3, 1)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a matrix of size 3-by-3.');
check_err(@() f33([]), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a matrix of size 3-by-3.');
check_err(@() f2c(ones(2, 2, 2)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a matrix with 2 rows.');
check_err(@() fx3(ones(2, 2)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a matrix with 3 columns.');
check_err(@() fcc(ones(2, 2, 2)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a matrix.');
check_err(@() fccc(ones(2, 2, 2, 2)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be a 3-dimensional array.');
check_err(@() f234(ones(2, 3)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be an array of size 2-by-3-by-4.');
check_err(@() f23c(ones(2, 2)), 'MATLAB:validation:IncompatibleSize', ...
  'Invalid argument at position 1. Value must be an array of size 2-by-3-by-* where * indicates a dimension of any size.');
assert(isequal(size(f23c(ones(2, 3, 4))), [2 3 4]));

% --- defaults and missing arguments ---
assert(fdefref(2) == 20);
assert(fdefref(2, 3) == 3);
check_err(@() fdefbad(), 'MATLAB:validators:mustBePositive', ...
  'Invalid default value for argument ''x''. Value must be positive.');
assert(fdefbad(4) == 4);
check_err(@() ftwo(), 'MATLAB:minrhs', ...
  'Invalid argument list. Function requires 2 more input(s).');
check_err(@() ftwo(1), 'MATLAB:minrhs', ...
  'Invalid argument list. Function requires 1 more input(s).');
check_err(@() fopt(), 'MATLAB:minrhs', ...
  'Invalid argument list. Function requires 1 more input(s).');
check_err(@() fpos(1, 2), 'MATLAB:TooManyInputs', 'Too many input arguments.');
assert(fnargin() == 0);
assert(fnargin(5) == 1);

disp('SUCCESS')

function check_err(f, id, msg)
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('identifier: got [%s], expected [%s]', e.identifier, id));
    assert(strcmp(e.message, msg), sprintf('message: got [%s], expected [%s]', e.message, msg));
    return;
  end
  error('expected an error: %s', msg);
end

function x = fpos(x)
  arguments
    x (1,1) double {mustBePositive}
  end
end

function y = fmulti(x, y)
  arguments
    x {mustBePositive}
    y {mustBePositive}
  end
end

function x = frange(x)
  arguments
    x {mustBeInRange(x, 0, 10)}
  end
end

function x = fmember(x)
  arguments
    x {mustBeMember(x, {'a', 'b'})}
  end
end

function x = fnumeric(x)
  arguments
    x {mustBeNumeric, mustBePositive}
  end
end

function y = fless(x, y)
  arguments
    x
    y {mustBeLessThan(y, x)}
  end
end

function x = fuserval(x)
  arguments
    x {myval}
  end
end

function myval(v)
  error('my:id', 'custom message %d', v);
end

function x = fuserval2(x)
  arguments
    x {myval2(x, 5)}
  end
end

function myval2(v, k)
  if v < k
    error('Value too small');
  end
end

function x = fdbl(x)
  arguments
    x double
  end
end

function x = fchar(x)
  arguments
    x char
  end
end

function x = fstr(x)
  arguments
    x string
  end
end

function x = flog(x)
  arguments
    x logical
  end
end

function x = fcell(x)
  arguments
    x cell
  end
end

function x = fstruct(x)
  arguments
    x struct
  end
end

function x = ffh(x)
  arguments
    x function_handle
  end
end

function x = fc11(x)
  arguments
    x (1,1) char
  end
end

function x = fs12(x)
  arguments
    x (1,2) string
  end
end

function x = frow(x)
  arguments
    x (1,:)
  end
end

function x = fcol(x)
  arguments
    x (:,1)
  end
end

function x = f13(x)
  arguments
    x (1,3)
  end
end

function x = f31(x)
  arguments
    x (3,1)
  end
end

function x = f33(x)
  arguments
    x (3,3) double
  end
end

function x = fx3(x)
  arguments
    x (:,3)
  end
end

function x = f2c(x)
  arguments
    x (2,:)
  end
end

function x = fcc(x)
  arguments
    x (:,:)
  end
end

function x = fccc(x)
  arguments
    x (:,:,:)
  end
end

function x = f234(x)
  arguments
    x (2,3,4)
  end
end

function x = f23c(x)
  arguments
    x (2,3,:)
  end
end

function y = fdefref(x, y)
  arguments
    x
    y = x * 10
  end
end

function x = fdefbad(x)
  arguments
    x {mustBePositive} = -1
  end
end

function r = ftwo(a, b)
  arguments
    a
    b
  end
  r = a + b;
end

function r = fopt(a, b)
  arguments
    a
    b = 1
  end
  r = a + b;
end

function r = fnargin(a)
  arguments
    a = 3
  end
  r = nargin + 0 * a;
end
