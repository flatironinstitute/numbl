% zeros/ones/nan/inf/eye/linspace/logspace: class-name and 'like'
% arguments, complex endpoints, and MATLAB's errors.

assert(isequal(zeros(2, 'double'), [0 0; 0 0]));
assert(isequal(zeros([2 3], 'double'), zeros(2, 3)));
assert(isequal(size(zeros(2, 3, 'like', 1)), [2 3]));
assert(isequal(zeros(2, "like", 1), [0 0; 0 0]));
assert(isequal(zeros(2, 'Like', 1), [0 0; 0 0]));
assert(zeros('double') == 0);
assert(isequal(size(nan(2, 'single')), [2 2]));
assert(isequal(size(zeros([])), [0 0]));
assert(isequal(size(zeros(-1)), [0 0]));
assert(isequal(eye([2 3]), [1 0 0; 0 1 0]));
assert(isequal(size(eye([])), [0 0]));

assert(isequal(linspace(1+1i, 2, 3), [1+1i 1.5+0.5i 2]));
assert(isequal(linspace('a', 'c', 3), [97 98 99]));
assert(isequal(linspace(1, 2, 2.5), [1 2]));
assert(isequal(linspace(1, 2, true), 2));
assert(isequal(linspace(1, 2, "3"), [1 1.5 2]));
assert(isequal(size(linspace(1, 2, 0)), [1 0]));
assert(isequal(logspace(1, 2, 2.5), [10 100]));
y = logspace(1+1i, 2, 3);
assert(abs(y(1) - (-6.68201510190313+7.43980336957493i)) < 1e-10);
assert(abs(y(2) - (12.880188077231+28.8808025355106i)) < 1e-10);
assert(y(3) == 100);

check_err(@() zeros(1+2i), 'MATLAB:obsoleteComplexInput', 'Size inputs must be real.');
check_err(@() zeros({1}), 'MATLAB:zeros:invalidNumericInput', 'Size inputs must be numeric.');
check_err(@() zeros(struct()), 'MATLAB:zeros:invalidNumericInput', 'Size inputs must be numeric.');
check_err(@() zeros('a'), 'MATLAB:zeros:invalidStringInput', 'Class name must be a class that supports ZEROS, such as "double" or "single".');
check_err(@() ones(2, 'foo'), 'MATLAB:ones:invalidStringInput', 'Class name must be a class that supports ONES, such as "double" or "single".');
check_err(@() zeros(2, 'double', 3), 'MATLAB:zeros:invalidNumericInput', 'Size inputs must be numeric.');
check_err(@() zeros(2, 'like'), 'MATLAB:zeros:invalidStringInput', 'Class name must be a class that supports ZEROS, such as "double" or "single".');
check_err(@() zeros(2, 'like', {1}), 'MATLAB:zeros:invalidInputClass', 'Argument following "like" must be a variable with a data type that supports ZEROS, such as a double or single variable.');
check_err(@() nan(2, 'int8'), 'MATLAB:NaN:invalidClassName', 'Class name must be a class that supports NAN, such as "double" or "single".');
check_err(@() inf(2, 'uint8'), 'MATLAB:Inf:invalidClassName', 'Class name must be a class that supports INF, such as "double" or "single".');
check_err(@() eye(2, 'foo'), 'MATLAB:eye:invalidStringInput', 'Class name must be a class that supports EYE, such as "single" or "double".');
check_err(@() eye(2, 3, 4), 'MATLAB:eye:unsupportedNd', 'N-dimensional arrays are not supported.');
check_err(@() eye([2 3 4]), 'MATLAB:eye:unsupportedNd', 'N-dimensional arrays are not supported.');
check_err(@() eye(1+1i), 'MATLAB:obsoleteComplexInput', 'Size inputs must be real.');
check_err(@() eye({2}), 'MATLAB:eye:invalidNumericInput', 'Size inputs must be numeric.');
check_err(@() zeros(2, [3 4]), 'MATLAB:NonScalarInput', 'Size inputs must be scalar.');
check_err(@() linspace(1, 2, [3 4]), 'MATLAB:linspace:scalarInputs', 'Inputs must be scalars.');
check_err(@() linspace([1; 2], [3; 4], 3), 'MATLAB:linspace:scalarInputs', 'Inputs must be scalars.');
check_err(@() linspace([], 2, 3), 'MATLAB:linspace:scalarInputs', 'Inputs must be scalars.');
check_err(@() linspace({1}, 2, 3), 'MATLAB:math:mustBeNumericCharOrLogical', 'Invalid data type. Argument must be numeric, char, or logical.');
check_err(@() linspace(1, 2, 1+1i), 'MATLAB:colon:operandsNotRealScalar', 'Colon operands must be real scalars.');
check_err(@() logspace(1, 2, [3 4]), 'MATLAB:linspace:scalarInputs', 'Inputs must be scalars.');

% arity
check_err(@() linspace(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() linspace(1, 2, 3, 4), 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err(@() logspace(1), 'MATLAB:minrhs', 'Not enough input arguments.');
assert(isequal(eye(2, 3, 1, 1), [1 0 0; 0 1 0]));
check_err(@() nout2(@zeros), 'MATLAB:maxlhs', 'Too many output arguments.');
check_err(@() nout2(@linspace), 'MATLAB:TooManyOutputs', 'Too many output arguments.');
assert(nargin(@zeros) == -1);
assert(nargin(@linspace) == 3);

disp('SUCCESS');

function nout2(f)
  [a, b] = f(1, 2);
end

function check_err(f, id, msg)
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('expected %s, got %s', id, e.identifier));
    assert(strcmp(e.message, msg), sprintf('expected "%s", got "%s"', msg, e.message));
    return;
  end
  error('expected error %s', id);
end
