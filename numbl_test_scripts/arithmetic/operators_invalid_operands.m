% Arithmetic, relational and logical operators reject cells, structs,
% function handles, dictionaries and string-valued negation with MATLAB's
% error, instead of returning a value. Transposes of cells and structs
% work; of a function handle they error.

s.a = 1;
f = @sin;
d = dictionary(1, 2);
c = {1};

% Each expression must raise MATLAB:math:mustBeNumericCharOrLogical.
exprs = {'~c', '~s', '~f', '~d', '~"ab"', ...
  '-c', '-s', '-f', '-d', '-"ab"', '+c', '+s', '+f', ...
  'c + 1', '1 + c', 's + 1', 'f + 1', 'd + 1', 'c - 1', 'c * 2', 'c / 2', ...
  'c \ 2', 'c .* 2', 'c ./ 2', 'c .\ 2', 'c ^ 2', 'c .^ 2', ...
  'c == 1', 'c ~= 1', 'c < 2', 's <= 1', 'f > 1', 'f == f', 's == s', ...
  'c & 1', '1 & c', 'c | 1', 's & 1', 'f | 1', 'd & 1', '"a" & 1', ...
  'not(c)', 'and(c, 1)', 'uminus(c)', 'plus(c, 1)'};
for i = 1:numel(exprs)
  threw = false;
  try
    eval([exprs{i} ';']);
  catch err
    threw = true;
    assert(strcmp(err.identifier, 'MATLAB:math:mustBeNumericCharOrLogical'), ...
      sprintf('%s: identifier %s', exprs{i}, err.identifier));
    assert(strcmp(err.message, ...
      'Invalid data type. Argument must be numeric, char, or logical.'), ...
      sprintf('%s: message %s', exprs{i}, err.message));
  end
  assert(threw, sprintf('%s did not error', exprs{i}));
end

% Transposing a function handle is an error.
threw = false;
try
  g = f';
catch err
  threw = true;
  assert(strcmp(err.identifier, 'MATLAB:transpose:unsupportedType'));
  assert(strcmp(err.message, 'Data type not supported.'));
end
assert(threw);

% Transposing cells, structs and dictionaries is fine.
c2 = {1, 2}';
assert(isequal(size(c2), [2 1]));
s2 = s';
assert(s2.a == 1);
d2 = d';
assert(isa(d2, 'dictionary'));

% ~ on char and logical values still works.
assert(isequal(~'ab', [false false]));
assert(isequal(~[true false], [false true]));
assert(isequal(~char([0 65]), [true false]));

% ~ on a complex value is an error.
threw = false;
try
  z = ~(1+2i);
catch err
  threw = true;
  assert(strcmp(err.identifier, 'MATLAB:Not:operandsNotReal'));
end
assert(threw);

% Unary plus turns char and logical into double.
assert(isequal(+'a', 97) && strcmp(class(+'a'), 'double'));
assert(strcmp(class(+true), 'double'));
assert(strcmp(class(+[true false]), 'double'));

% `x ./ 2` as a statement is an expression, not command syntax.
x = [4 6];
x ./ 2;
assert(isequal(ans, [2 3]));
x .\ 8;
assert(isequal(ans, [2 4/3]));

disp('SUCCESS')
