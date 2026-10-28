% Arity, argument checks and fixed edge cases for array-extras and
% type-constructor builtins (colon, triu/tril, magic, bit ops, coordinate
% transforms, nchoosek, double/int*/logical/complex/cell/struct/...).

%% Arity errors
check_err(@() colon(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() colon(1, 2, 3, 4), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() triu(1, 2, 3), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() magic(), 'MATLAB:minrhs', '');
check_err(@() magic(3, 4), 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err(@() bitand(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() bitset(1, 2, 3, 'uint8', 5), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() nchoosek(5), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() cart2pol(1, 2, 3, 4), 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err(@() double(), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() int8(1, 2), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() logical(1, 2), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() complex(1, 2, 3), 'MATLAB:maxrhs', 'Too many input arguments.');
check_err(@() cell(), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() isfield(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() idivide(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() typecast(1), 'MATLAB:minrhs', 'Not enough input arguments.');
check_err(@() func2str(), 'MATLAB:minrhs', 'Not enough input arguments.');
try
  [a, b] = double(1); %#ok<ASGLU>
  error('expected an error');
catch e
  assert(strcmp(e.identifier, 'MATLAB:maxlhs'));
end
try
  [a, b] = toeplitz(1); %#ok<ASGLU>
  error('expected an error');
catch e
  assert(strcmp(e.identifier, 'MATLAB:TooManyOutputs'));
end
assert(nargin(@colon) == 3);
assert(nargin(@bitset) == 4);
assert(nargin(@complex) == 2);

%% colon
assert(isequal(colon(1, 3), [1 2 3]));
assert(isequal(colon(1, 0.5, 2), [1 1.5 2]));
c = colon('a', 'c');
assert(ischar(c) && strcmp(c, 'abc'));
check_err(@() colon(1+2i, 3), 'MATLAB:colon:operandsNotRealScalar', 'Colon operands must be real scalars.');
check_err(@() colon([1 2], 3), 'MATLAB:colon:operandsNotRealScalar', 'Colon operands must be real scalars.');
check_err(@() colon({1}, 3), 'MATLAB:colon:inputsMustBeNumericCharLogical', 'Inputs must be numeric, char, or logical.');

%% triu / tril
check_err(@() triu(magic(3), 1.5), 'MATLAB:triu:kthDiagInputNotInteger', 'K-th diagonal input must be an integer scalar.');
check_err(@() tril(magic(3), 1i), 'MATLAB:tril:kthDiagInputNotInteger', 'K-th diagonal input must be an integer scalar.');
check_err(@() triu(ones(2, 2, 2)), 'MATLAB:triu:firstInputMustBe2D', 'First input must be 2D.');
check_err(@() triu({1}), 'MATLAB:math:mustBeNumericCharOrLogical', 'Invalid data type. Argument must be numeric, char, or logical.');
assert(triu(2+3i) == 2+3i);
assert(tril(2+3i, -1) == 0);

%% magic for small and non-integer orders
assert(isequal(magic(0), []));
assert(isequal(magic(-1), []));
assert(magic(1) == 1);
assert(isequal(magic(2), [1 3; 4 2]));
assert(isequal(magic(3.7), magic(3)));

%% toeplitz
check_err(@() toeplitz({1}), 'MATLAB:toeplitz:nonNumericInputs', 'Inputs must be numeric.');

%% bit ops reject complex and non-numeric input
check_err(@() bitand(1+2i, 1), 'MATLAB:Bitand:complexInputs', 'Inputs must be real.');
check_err(@() bitor([1 2]*1i, 1), 'MATLAB:Bitor:complexInputs', 'Inputs must be real.');
check_err(@() bitxor(1, 2i), 'MATLAB:bitxor:complexInputs', 'Inputs must be real.');
check_err(@() bitshift(1+1i, 1), 'MATLAB:bitshift:complexArgs', 'Inputs must be real.');
check_err(@() bitget(1i, 1), 'MATLAB:bitget:complexArgs', 'Inputs must be real.');
check_err(@() bitset(5, 2, 1i), 'MATLAB:bitset:complexArgs', 'Inputs must be real.');
check_err(@() bitand('a', 1), 'MATLAB:Bitand:operandsNotNumeric', 'Operands to BIT Ops must be numeric.');
check_err(@() bitget({1}, 1), 'MATLAB:bitget:numericArg', 'Inputs must be numeric.');
check_err(@() bitand(5, 3, 'foo'), 'MATLAB:Bitand:unknownTypeString', 'ASSUMEDTYPE must be an integer type name.');
check_err(@() bitshift(1, 2, 3), 'MATLAB:bitshift:unknownTypeString', 'ASSUMEDTYPE must be an integer type name.');
assert(bitand(12, 10) == 8);
assert(bitset(5, 2, 2) == 7);
assert(bitset(5, 2, 0) == 5);

%% coordinate transforms
check_err(@() cart2pol(1+1i, 1), 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err(@() cart2sph(1, 1, 1i), 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err(@() cart2pol('a', 1), 'MATLAB:math:mustBeFloat', 'Invalid data type. Argument must be double or single.');
check_err(@() cart2pol([1 2], [1 2 3]), 'MATLAB:sizeDimensionsMustMatch', 'Arrays have incompatible sizes for this operation.');
[th, r] = cart2pol([1 2], [1; 2]);
assert(isequal(size(th), [2 2]));
assert(abs(th(2, 1) - atan2(2, 1)) < 1e-12 && abs(r(2, 2) - sqrt(8)) < 1e-12);
[th, r, z] = cart2pol(1, 1, [5 6]);
assert(abs(th - pi/4) < 1e-12 && abs(r - sqrt(2)) < 1e-12 && isequal(z, [5 6]));
[x, y] = pol2cart(0, [1 2]);
assert(isequal(x, [1 2]) && isequal(y, [0 0]));

%% nchoosek
check_err(@() nchoosek(5, 2.5), 'MATLAB:nchoosek:InvalidArg2', 'The second input has to be a non-negative integer.');
check_err(@() nchoosek(5, -1), 'MATLAB:nchoosek:InvalidArg2', 'The second input has to be a non-negative integer.');
assert(nchoosek(5, 2) == 10);
C = nchoosek([1i 2 3], 2);
assert(isequal(C, [1i 2; 1i 3; 2 3]));

%% double keeps complex values and sparse storage
assert(double(1+2i) == 1+2i);
assert(issparse(double(sparse(1))));
check_err(@() double({1}), 'MATLAB:invalidConversion', 'Conversion to double from cell is not possible.');
check_err(@() double(struct()), 'MATLAB:invalidConversion', 'Conversion to double from struct is not possible.');
check_err(@() double(@sin), 'MATLAB:invalidConversion', 'Conversion to double from function_handle is not possible.');
check_err(@() single({1}), 'MATLAB:invalidConversion', 'Conversion to single from cell is not possible.');

%% integer constructors keep the imaginary part
z = int8(1.6+2.4i);
assert(real(z) == 2 && imag(z) == 2);
z = uint8(-3+300i);
assert(real(z) == 0 && imag(z) == 255);
z = int8([1.6+2.4i 3]);
assert(isequal(real(z), [2 3]) && isequal(imag(z), [2 0]));
check_err(@() int8({1}), 'MATLAB:invalidConversion', 'Conversion to int8 from cell is not possible.');
check_err(@() int16("5"), 'MATLAB:invalidConversion', 'Conversion to int16 from string is not possible.');

%% logical
assert(islogical(logical('a')) && logical('a'));
assert(isequal(logical('ab'), [true true]));
check_err(@() logical(1i), 'MATLAB:nologicalcomplex', 'Complex values cannot be converted to logicals.');
check_err(@() logical(NaN), 'MATLAB:nologicalnan', 'NaN values cannot be converted to logicals.');
check_err(@() logical([1 NaN]), 'MATLAB:nologicalnan', 'NaN values cannot be converted to logicals.');
check_err(@() logical({1}), 'MATLAB:invalidConversion', 'Conversion to logical from cell is not possible.');

%% complex
check_err(@() complex(1+2i, 1), 'MATLAB:complex:realPartInputNonNumericOrNonReal', 'Input for real part must be a real numeric value.');
check_err(@() complex(1, 2i), 'MATLAB:complex:imagPartInputNonNumericOrNonReal', 'Input for imaginary part must be a real numeric value.');
check_err(@() complex([1+1i 2], 1), 'MATLAB:complex:realPartInputNonNumericOrNonReal', 'Input for real part must be a real numeric value.');
check_err(@() complex([1 2], [1 2 3]), 'MATLAB:complex:inputSizeMismatch', 'Input arrays must have the same size.');
check_err(@() complex({1}), 'MATLAB:complex:expectsNumeric', 'First input must be numeric.');
assert(complex(1+2i) == 1+2i);
assert(complex('a') == 97);
assert(complex('a', 1) == 97+1i);
assert(isequal(complex([1 2], 3), [1+3i 2+3i]));
assert(isequal(complex([1 2], ones(1, 1)), [1+1i 2+1i]));

%% cell sizes
check_err(@() cell('a'), 'MATLAB:invalidConversion', 'Conversion to cell from char is not possible.');
check_err(@() cell(1i), 'MATLAB:obsoleteComplexInput', 'Size inputs must be real.');
assert(isequal(size(cell(-1)), [0 0]));
assert(isequal(size(cell([])), [0 0]));
assert(isequal(size(cell(2, 3, 1)), [2 3]));

%% struct
s = struct('a', 1);
assert(isequal(struct(s), s));
check_err(@() struct(1), 'MATLAB:invalidConversion', 'Conversion to struct from double is not possible.');
check_err(@() struct('a', 1, 'b'), 'MATLAB:StructConversion:NonPairedArgs', 'Field and value input arguments must come in pairs.');
check_err(@() struct(1, 2), 'MATLAB:Cell2Struct:NonStringFieldNames', 'Field names must be non-empty character vectors or string scalars.');
check_err(@() struct('a', {1, 2}, 'b', {1, 2, 3}), 'MATLAB:StructConversion:DimensionMismatch', 'Array dimensions of input ''4'' must match those of input ''2'', or be scalar.');
t = struct('a', {1, 2}, 'b', {3});
assert(numel(t) == 2 && t(2).a == 2 && t(1).b == 3 && t(2).b == 3);

%% typecast
check_err(@() typecast(1, 'foo'), 'MATLAB:typecast:unsupportedClass', 'Data conversion from ''double'' to ''foo'' is not supported.');
check_err(@() typecast(1, 1), 'MATLAB:typecast:invalidSyntax', 'Second argument must be either a character row vector or string scalar.');
check_err(@() typecast({1}, 'uint8'), 'MATLAB:typecast:mustBeNumericCharOrLogical', 'First argument must be numeric, char, or logical.');

%% jsonencode options
assert(strcmp(jsonencode([1 NaN]), '[1,null]'));
assert(strcmp(jsonencode([1 Inf -Inf NaN], 'ConvertInfAndNaN', false), '[1,Infinity,-Infinity,NaN]'));
assert(strcmp(jsonencode(struct('a', NaN), 'ConvertInfAndNaN', false), '{"a":NaN}'));
assert(strcmp(jsonencode(1, 'PrettyPrint', false), '1'));
check_err(@() jsonencode(1, 'foo', 1), 'MATLAB:json:UnmatchedParameter', 'Name argument must be ''ConvertInfAndNaN'' or ''PrettyPrint''.');
check_err(@() jsonencode(1, 'PrettyPrint'), 'MATLAB:json:ParamMissingValue', 'Name ''PrettyPrint'' requires a value.');

%% idivide
check_err(@() idivide(int8(7), int8(2), 'foo'), 'MATLAB:idivide:InvalidRoundingOption', 'Unrecognized rounding option.');
assert(idivide(int8(7), int8(2), "floor") == 3);

%% func2str / deal
check_err(@() func2str(1), 'MATLAB:UndefinedFunction', 'Undefined function ''func2str'' for input arguments of type ''double''.');
check_err(@() func2str('sin'), 'MATLAB:UndefinedFunction', 'Undefined function ''func2str'' for input arguments of type ''char''.');
[p, q] = deal(7);
assert(p == 7 && q == 7);
[p, q] = deal(1, 2);
assert(p == 1 && q == 2);
try
  [p, q] = deal(1, 2, 3); %#ok<ASGLU>
  error('expected an error');
catch e
  assert(strcmp(e.identifier, 'MATLAB:deal:narginNargoutMismatch'));
  assert(strcmp(e.message, 'The number of outputs should match the number of inputs.'));
end

disp('SUCCESS');

function check_err(f, id, msg)
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('expected id %s, got %s (%s)', id, e.identifier, e.message));
    if ~isempty(msg)
      assert(strcmp(e.message, msg), sprintf('expected message "%s", got "%s"', msg, e.message));
    end
    return;
  end
  error('expected error %s', id);
end
