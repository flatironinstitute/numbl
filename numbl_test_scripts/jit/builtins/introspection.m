% ===== isnumeric =====

function test_isnumeric()
assert(isequal(isnumeric(5), true));
assert(isequal(isnumeric(0), true));
assert(isequal(isnumeric(-3.14), true));
assert(isequal(isnumeric(true), false));
assert(isequal(isnumeric(false), false));
end

function test_isnumeric_complex()
assert(isequal(isnumeric(1+2i), true));
assert(isequal(isnumeric(3i), true));
end

function test_isnumeric_tensor()
assert(isequal(isnumeric([1 2 3]), true));
assert(isequal(isnumeric([1 2; 3 4]), true));
end

function test_isnumeric_complex_tensor()
assert(isequal(isnumeric([1+2i 3+4i]), true));
end

% ===== isfloat =====

function test_isfloat()
assert(isequal(isfloat(5), true));
assert(isequal(isfloat(0.5), true));
assert(isequal(isfloat(true), false));
assert(isequal(isfloat(false), false));
end

function test_isfloat_complex()
assert(isequal(isfloat(1+2i), true));
end

function test_isfloat_tensor()
assert(isequal(isfloat([1 2 3]), true));
end

function test_isfloat_complex_tensor()
assert(isequal(isfloat([1+2i 3]), true));
end

% ===== isinteger =====

function test_isinteger()
assert(isequal(isinteger(5), false));
assert(isequal(isinteger(true), false));
assert(isequal(isinteger([1 2]), false));
assert(isequal(isinteger(1+2i), false));
end

% ===== islogical =====

function test_islogical()
assert(isequal(islogical(true), true));
assert(isequal(islogical(false), true));
assert(isequal(islogical(5), false));
assert(isequal(islogical(0), false));
assert(isequal(islogical(1+2i), false));
end

% ===== ischar =====

function test_ischar()
assert(isequal(ischar(5), false));
assert(isequal(ischar(true), false));
assert(isequal(ischar([1 2]), false));
assert(isequal(ischar(1+2i), false));
end

% ===== isstring =====

function test_isstring()
assert(isequal(isstring(5), false));
assert(isequal(isstring(true), false));
assert(isequal(isstring([1 2]), false));
end

% ===== iscell =====

function test_iscell()
assert(isequal(iscell(5), false));
assert(isequal(iscell(true), false));
assert(isequal(iscell([1 2]), false));
end

% ===== isstruct =====

function test_isstruct()
assert(isequal(isstruct(5), false));
assert(isequal(isstruct(true), false));
assert(isequal(isstruct([1 2]), false));
end

% ===== issparse =====

function test_issparse()
assert(isequal(issparse(5), false));
assert(isequal(issparse(true), false));
assert(isequal(issparse([1 2]), false));
end

% ===== isscalar =====

function test_isscalar()
assert(isequal(isscalar(5), true));
assert(isequal(isscalar(true), true));
assert(isequal(isscalar(1+2i), true));
assert(isequal(isscalar([1 2 3]), false));
assert(isequal(isscalar([1; 2]), false));
assert(isequal(isscalar([1 2; 3 4]), false));
end

% ===== isempty =====

function test_isempty()
assert(isequal(isempty(5), false));
assert(isequal(isempty(0), false));
assert(isequal(isempty(true), false));
assert(isequal(isempty(1+2i), false));
assert(isequal(isempty([]), true));
assert(isequal(isempty([1 2 3]), false));
end

% ===== isvector =====

function test_isvector()
assert(isequal(isvector(5), true));
assert(isequal(isvector(true), true));
assert(isequal(isvector([1 2 3]), true));
assert(isequal(isvector([1; 2; 3]), true));
assert(isequal(isvector([1 2; 3 4]), false));
end

% ===== isrow =====

function test_isrow()
assert(isequal(isrow(5), true));
assert(isequal(isrow([1 2 3]), true));
assert(isequal(isrow([1; 2; 3]), false));
assert(isequal(isrow([1 2; 3 4]), false));
end

% ===== iscolumn =====

function test_iscolumn()
assert(isequal(iscolumn(5), true));
assert(isequal(iscolumn([1; 2; 3]), true));
assert(isequal(iscolumn([1 2 3]), false));
assert(isequal(iscolumn([1 2; 3 4]), false));
end

% ===== ismatrix =====

function test_ismatrix()
assert(isequal(ismatrix(5), true));
assert(isequal(ismatrix(true), true));
assert(isequal(ismatrix([1 2 3]), true));
assert(isequal(ismatrix([1 2; 3 4]), true));
end

% ===== numel =====

function test_numel()
assert(isequal(numel(5), 1));
assert(isequal(numel(true), 1));
assert(isequal(numel(1+2i), 1));
assert(isequal(numel([1 2 3]), 3));
assert(isequal(numel([1 2; 3 4]), 4));
end

function test_numel_complex_tensor()
assert(isequal(numel([1+2i 3+4i 5]), 3));
end

% ===== length =====

function test_length()
assert(isequal(length(5), 1));
assert(isequal(length(true), 1));
assert(isequal(length(1+2i), 1));
assert(isequal(length([1 2 3]), 3));
assert(isequal(length([1; 2; 3]), 3));
assert(isequal(length([1 2; 3 4; 5 6]), 3));
end

% ===== ndims =====

function test_ndims()
assert(isequal(ndims(5), 2));
assert(isequal(ndims(true), 2));
assert(isequal(ndims([1 2 3]), 2));
assert(isequal(ndims([1 2; 3 4]), 2));
end

% ===== size =====

function test_size_with_dim()
assert(isequal(size(5, 1), 1));
assert(isequal(size(5, 2), 1));
assert(isequal(size([1 2 3], 1), 1));
assert(isequal(size([1 2 3], 2), 3));
assert(isequal(size([1 2; 3 4], 1), 2));
assert(isequal(size([1 2; 3 4], 2), 2));
end

function test_size_vector()
s = size([1 2 3]);
assert(isequal(s, [1 3]));
s2 = size([1 2; 3 4]);
assert(isequal(s2, [2 2]));
s3 = size(5);
assert(isequal(s3, [1 1]));
end

% ===== class =====

function test_class_number()
assert(isequal(class(5), 'double'));
assert(isequal(class(0), 'double'));
assert(isequal(class(-3.14), 'double'));
end

function test_class_logical()
assert(isequal(class(true), 'logical'));
assert(isequal(class(false), 'logical'));
end

function test_class_complex()
assert(isequal(class(1+2i), 'double'));
assert(isequal(class(3i), 'double'));
end

function test_class_tensor()
assert(isequal(class([1 2 3]), 'double'));
end

function test_class_complex_tensor()
assert(isequal(class([1+2i 3+4i]), 'double'));
end

% ===== predicate arithmetic =====

function test_predicate_arithmetic()
assert(isequal(isnumeric(5) + 1, 2));
assert(isequal(isfloat(5) + 1, 2));
assert(isequal(isscalar(5) + 1, 2));
assert(isequal(isempty([]) + 1, 2));
end

% ===== predicate comparison =====

function test_predicate_comparison()
assert(isnumeric(5) == 1);
assert(isfloat(5) == 1);
assert(islogical(true) == 1);
assert(isinteger(5) == 0);
end

% ===== Call tests =====

%!jit
test_isnumeric();
%!jit
test_isnumeric_complex();
%!jit
test_isnumeric_tensor();
%!jit
test_isnumeric_complex_tensor();
%!jit
test_isfloat();
%!jit
test_isfloat_complex();
%!jit
test_isfloat_tensor();
%!jit
test_isfloat_complex_tensor();
%!jit
test_isinteger();
%!jit
test_islogical();
%!jit
test_ischar();
%!jit
test_isstring();
%!jit
test_iscell();
%!jit
test_isstruct();
%!jit
test_issparse();
%!jit
test_isscalar();
%!jit
test_isempty();
%!jit
test_isvector();
%!jit
test_isrow();
%!jit
test_iscolumn();
%!jit
test_ismatrix();
%!jit
test_numel();
%!jit
test_numel_complex_tensor();
%!jit
test_length();
%!jit
test_ndims();
%!jit
test_size_with_dim();
%!jit
test_size_vector();
%!jit
test_class_number();
%!jit
test_class_logical();
%!jit
test_class_complex();
%!jit
test_class_tensor();
%!jit
test_class_complex_tensor();
%!jit
test_predicate_arithmetic();
%!jit
test_predicate_comparison();

disp('SUCCESS');
