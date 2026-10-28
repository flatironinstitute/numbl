% mustBe* validator functions: accepted values, and MATLAB's messages and
% identifiers for rejected ones. Verified against MATLAB R2025b.

mustBeNumeric(42);
mustBeNumeric([1 2 3]);
mustBeNumeric(1 + 2i);
check_err(@() mustBeNumeric(true), 'mustBeNumeric', 'Value must be numeric.');
check_err(@() mustBeNumeric('a'), 'mustBeNumeric', 'Value must be numeric.');
mustBeNumericOrLogical(true);
check_err(@() mustBeNumericOrLogical('a'), 'mustBeNumericOrLogical', ...
  'Value must be numeric or logical.');

mustBeInteger(5);
mustBeInteger([1 2 3]);
check_err(@() mustBeInteger(1.5), 'mustBeInteger', 'Value must be integer.');
check_err(@() mustBeInteger('a'), 'mustBeNumericOrLogical', ...
  'Value must be numeric or logical.');

mustBePositive(1);
mustBePositive([1 2 3]);
mustBePositive([]);
mustBePositive(true);
check_err(@() mustBePositive([-1 2]), 'mustBePositive', 'Value must be positive.');
check_err(@() mustBePositive(NaN), 'mustBePositive', 'Value must be positive.');
check_err(@() mustBePositive('a'), 'mustBeNumericOrLogical', ...
  'Value must be numeric or logical.');
check_err(@() mustBePositive({1}), 'mustBeNumericOrLogical', ...
  'Value must be numeric or logical.');
mustBeNonnegative([0 1]);
check_err(@() mustBeNonnegative(-1), 'mustBeNonnegative', 'Value must be nonnegative.');
mustBeNonpositive([0 -1]);
check_err(@() mustBeNonpositive(1), 'mustBeNonpositive', 'Value must not be positive.');
mustBeNegative(-1);
check_err(@() mustBeNegative(1), 'mustBeNegative', 'Value must be negative.');
mustBeNonzero([-1 2 3]);
check_err(@() mustBeNonzero([1 0 3]), 'mustBeNonzero', 'Value must not be zero.');

mustBeFinite([1 2 3]);
mustBeFinite('a');
check_err(@() mustBeFinite(Inf), 'mustBeFinite', 'Value must be finite.');
check_err(@() mustBeFinite([1 NaN 3]), 'mustBeFinite', 'Value must be finite.');
check_err(@() mustBeNonNan(NaN), 'mustBeNonNan', 'Value must not be NaN.');
mustBeReal(1);
check_err(@() mustBeReal(1i), 'mustBeReal', 'Value must be real.');

mustBeNonempty('hello');
check_err(@() mustBeNonempty([]), 'mustBeNonempty', 'Value must not be empty.');
check_err(@() mustBeNonempty(''), 'mustBeNonempty', 'Value must not be empty.');
mustBeNonsparse(1);
check_err(@() mustBeNonsparse(sparse(1)), 'mustBeNonsparse', 'Value must not be sparse.');
mustBeScalarOrEmpty(42);
mustBeScalarOrEmpty([]);
check_err(@() mustBeScalarOrEmpty([1 2 3]), 'mustBeScalarOrEmpty', ...
  'Value must be scalar or empty.');

mustBeVector(42);
mustBeVector([1 2 3]);
mustBeVector([1; 2; 3]);
mustBeVector(zeros(1, 0));
mustBeVector([], 'allow-all-empties');
check_err(@() mustBeVector([1 2; 3 4]), 'mustBeVector', ...
  'Value must be a 1-by-n vector or an n-by-1 vector.');
check_err(@() mustBeVector([]), 'mustBeVector', ...
  'Value must be a 1-by-n vector or an n-by-1 vector.');
mustBeRow([1 2]);
check_err(@() mustBeRow([1; 2]), 'mustBeRow', 'Value must be a row vector.');
mustBeColumn([1; 2]);
check_err(@() mustBeColumn([1 2]), 'mustBeColumn', 'Value must be a column vector.');
mustBeMatrix(ones(2));
check_err(@() mustBeMatrix(ones(2, 2, 2)), 'mustBeMatrix', 'Value must be a matrix.');

mustBeText({'a', 'b'});
mustBeText("a");
check_err(@() mustBeText(1), 'mustBeText', ...
  'Value must be a character vector, string array, or cell array of character vectors.');
mustBeTextScalar('');
check_err(@() mustBeTextScalar(["a", "b"]), 'mustBeTextScalar', ...
  'Value must be a character vector or string scalar.');
check_err(@() mustBeNonzeroLengthText(''), 'mustBeNonzeroLengthText', ...
  'Value must be text with one or more characters.');
check_err(@() mustBeNonzeroLengthText(1), 'mustBeNonzeroLengthText', ...
  'Value must be a character vector, string array, or cell array of character vectors.');
mustBeValidVariableName('abc');
check_err(@() mustBeValidVariableName('1a'), 'mustBeValidVariableName', ...
  'These are not valid variable names: ''1a''.');

mustBeGreaterThan(3, 2);
check_err(@() mustBeGreaterThan(1, 2), 'mustBeGreaterThan', 'Value must be greater than 2.');
check_err(@() mustBeGreaterThan(1, 2.5), 'mustBeGreaterThan', 'Value must be greater than 2.5.');
check_err(@() mustBeGreaterThanOrEqual(1, 2), 'mustBeGreaterThanOrEqual', ...
  'Value must be greater than or equal to 2.');
check_err(@() mustBeLessThan(3, 2), 'mustBeLessThan', 'Value must be less than 2.');
check_err(@() mustBeLessThanOrEqual(3, 2), 'mustBeLessThanOrEqual', ...
  'Value must be less than or equal to 2.');
try
  mustBeLessThan(3, [1 2]);
  error('should have thrown');
catch e
  assert(strcmp(e.identifier, 'MATLAB:validatorUsage:nonScalarSecondInput'));
  assert(strcmp(e.message, 'Second input to function ''mustBeLessThan'' must be a scalar.'));
end

mustBeInRange(5, 1, 10);
mustBeInRange([2 5 8], 1, 10);
mustBeInRange(10, 0, 10);
check_err(@() mustBeInRange(15, 1, 10), 'mustBeInRange', ...
  'Value must be greater than or equal to 1, and less than or equal to 10.');
check_err(@() mustBeInRange(10, 0, 10, 'exclusive'), 'mustBeInRange', ...
  'Value must be greater than 0, and less than 10.');
check_err(@() mustBeInRange(0, 0, 10, 'exclude-lower'), 'mustBeInRange', ...
  'Value must be greater than 0, and less than or equal to 10.');
check_err(@() mustBeInRange(10, 0, 10, 'exclude-upper'), 'mustBeInRange', ...
  'Value must be greater than or equal to 0, and less than 10.');
check_err(@() mustBeInRange(0.5, 0.25, 0.4), 'mustBeInRange', ...
  'Value must be greater than or equal to 0.250000000000000, and less than or equal to 0.400000000000000.');
check_err(@() mustBeInRange('a', 0, 10), 'mustBeNumericOrLogical', ...
  'Value must be numeric or logical.');

mustBeMember(2, [1 2 3]);
mustBeMember('a', {'a', 'b'});
mustBeMember("b", ["a", "b"]);
check_err(@() mustBeMember(5, [1 2]), 'mustBeMember', ...
  sprintf('Value must be a member of this set:\n    1\n    2\n'));
check_err(@() mustBeMember([1 3], [1 2 4]), 'mustBeMember', ...
  sprintf('Value must be a member of this set:\n    1\n    2\n    4\n'));
check_err(@() mustBeMember('c', {'a', 'b'}), 'mustBeMember', ...
  sprintf('Value must be a member of this set:\n    ''a''\n    ''b''\n'));
check_err(@() mustBeMember("c", ["a", "b"]), 'mustBeMember', ...
  sprintf('Value must be a member of this set:\n    ''a''\n    ''b''\n'));

mustBeA(1, 'double');
mustBeA('x', {'char', 'string'});
check_err(@() mustBeA(1, 'char'), 'mustBeA', ...
  'Value must be one of these types: ''char''.');
check_err(@() mustBeA(1, {'char', 'cell'}), 'mustBeA', ...
  'Value must be one of these types: ''char'' or ''cell''.');
check_err(@() mustBeA(1, ["char", "string", "cell"]), 'mustBeA', ...
  'Value must be one of these types: ''char'', ''string'', or ''cell''.');

disp('SUCCESS')

function check_err(f, name, msg)
  id = ['MATLAB:validators:' name];
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('identifier: got [%s], expected [%s]', e.identifier, id));
    assert(strcmp(e.message, msg), sprintf('message: got [%s], expected [%s]', e.message, msg));
    return;
  end
  error('expected an error: %s', msg);
end
