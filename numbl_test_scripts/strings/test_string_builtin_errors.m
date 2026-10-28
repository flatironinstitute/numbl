% Arity and argument-type errors for string builtins (MATLAB's messages and
% identifiers), plus call forms fixed or added when they moved to the
% builtin() API.

% ── input/output count ──────────────────────────────────────────────────
check_err('upper()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('upper(''a'', ''b'')', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('[a, b] = upper(''a'')', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('strrep(''a'', ''b'')', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('strrep(''a'', ''b'', ''c'', ''d'')', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('strtrim(''a'', 1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('strcmp(''a'')', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('newline(1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('sscanf(''1'')', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('[a, b, c, d, e] = sscanf(''1'', ''%d'')', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('[a, b, c] = sprintf(''a'')', 'MATLAB:maxlhs', 'Too many output arguments.');
% M-file functions in MATLAB
check_err('reverse(''a'', ''b'')', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('num2str(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('[a, b] = num2str(1)', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('[a, b, c] = strtok(''a b'')', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('strjoin({''a''}, '' '', 1)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('[a, b, c] = strsplit(''a b'')', 'MATLAB:TooManyOutputs', 'Too many output arguments.');

% ── argument types ──────────────────────────────────────────────────────
mustBeText = 'MATLAB:string:MustBeCharCellArrayOrString';
check_err('reverse(5)', mustBeText, 'First argument must be text.');
check_err('erase(5, ''a'')', mustBeText, 'First argument must be text.');
check_err('contains(5, ''a'')', mustBeText, 'First argument must be text.');
check_err('strlength(5)', mustBeText, 'First argument must be text.');
check_err('split(5)', mustBeText, 'First argument must be text.');
check_err('join(5)', mustBeText, 'First argument must be text.');
check_err('strtrim(5)', 'MATLAB:strtrim:InputClass', ...
  'Input should be a string, character array, or a cell array of character arrays.');
check_err('erase(''abc'', 5)', 'MATLAB:string:MatchMustBeStringCharOrCellArrayOfChars', ...
  'Text to replace must be a text or pattern array.');
check_err('count(''abc'', 5)', 'MATLAB:string:MustBeTextOrPattern', ...
  'Search term must be a text or pattern array.');
check_err('regexprep(1, ''a'', ''b'')', 'MATLAB:REGEXP:invalidInputs', ...
  'The ''STRING'' input must be either a char row vector, a cell array of char row vectors, or a string array.');
check_err('isstrprop(''abc'', 5)', 'MATLAB:isstrprop:InvalidQualifierType', ...
  'Invalid data type in second argument.');
check_err('hex2dec(5)', 'MATLAB:hex2dec:InvalidCharacters', ...
  'Hexadecimal text must consist of characters 0-9 and A-F.');
check_err('bin2dec(5)', 'MATLAB:bin2dec:InputMustBeString', ...
  'Input argument must be a character vector, string, or cell array of character vectors.');
check_err('blanks(''a'')', 'MATLAB:repmat:nonNumericReplications', ...
  'Replication factors must be numeric.');
check_err('blanks([1 2])', 'MATLAB:NonScalarInput', 'Size inputs must be scalar.');
check_err('dec2bin({1})', 'MATLAB:dec2bin:InvalidDecimalArg', 'D must be numeric.');
check_err('dec2hex({1})', 'MATLAB:dec2hex:FirstArgIsInvalid', ...
  'First argument must contain finite integers.');
check_err('int2str({1})', 'MATLAB:math:mustBeNumericCharOrLogical', ...
  'Invalid data type. Argument must be numeric, char, or logical.');
check_err('num2str({1})', 'MATLAB:num2str:nonNumericInput', ...
  'Input to num2str must be numeric.');
check_err('num2str(5, {1})', 'MATLAB:num2str:invalidSecondArgument', ...
  'Second argument to num2str must be a character vector, string scalar, or numeric.');
check_err('mat2str({1})', 'MATLAB:mat2str:NumericInput', ...
  'Input matrix must be a numeric array, character array, or string array.');
check_err('char(struct())', 'MATLAB:invalidConversion', ...
  'Conversion to char from struct is not possible.');
check_err('char(true)', 'MATLAB:invalidConversion', ...
  'Conversion to char from logical is not possible.');
check_err('char(''ab'', {''c''})', 'MATLAB:character:InvalidInput', ...
  'Inputs must be character arrays.');
check_err('cellstr(5)', 'MATLAB:invalidConversion', ...
  'Conversion to cellstr from double is not possible.');
check_err('strings(''a'')', 'MATLAB:class:MustBeNumeric', ...
  'Array must be numeric or logical.');
check_err('sprintf(5)', 'MATLAB:badformat_mx', 'Invalid format.');
check_err('sscanf(5, ''%d'')', 'MATLAB:badstringarg1_mx', ...
  'First argument must be a text scalar.');
check_err('sscanf(''1'', 5)', 'MATLAB:badformat_mx', 'Invalid format.');
check_err('strip(''  a  '', ''foo'')', 'MATLAB:string:OptionMustBeOneOf', ...
  'Option must be ''left'', ''right'', or ''both'', or it must be exactly one character.');

% ── forms fixed or added ────────────────────────────────────────────────
% lower/upper return non-text input unchanged
assert(isequal(upper(5), 5));
assert(isequal(lower(true), true));

% IgnoreCase for contains and count
assert(contains('ABC', 'b', 'IgnoreCase', true));
assert(~contains('ABC', 'b', 'IgnoreCase', false));
assert(count('ABCabc', 'b', 'IgnoreCase', true) == 2);
assert(count('ABCabc', 'b') == 1);

% strip with a strip character, and on cell and string arrays
assert(strcmp(strip('xxaxx', 'x'), 'a'));
assert(strcmp(strip('xxaxx', 'both', 'x'), 'a'));
assert(strcmp(strip('xxaxx', 'left', 'x'), 'axx'));
assert(isequal(strip({' a ', 'b '}), {'a', 'b'}));
assert(isequal(strip(["  a", "b  "], 'left'), ["a", "b  "]));

% pad forms
assert(strcmp(pad('ab'), 'ab'));
assert(strcmp(pad('ab', 'left'), 'ab'));
assert(strcmp(pad('ab', 5, 'left', '*'), '***ab'));
assert(strcmp(pad('ab', 5, '*'), 'ab***'));
assert(strcmp(pad('ab', 5, 'both'), ' ab  '));
assert(strcmp(pad("ab", 4), "ab  "));

% char with no or several arguments
c = char();
assert(ischar(c) && isequal(size(c), [0 0]));
c = char('a', '', 'bcd');
assert(isequal(size(c), [3 3]));
assert(strcmp(c(1, :), 'a  ') && strcmp(c(2, :), '   ') && strcmp(c(3, :), 'bcd'));
assert(isequal(char(65, 'bc'), ['A '; 'bc']));

% string() is the empty string scalar
s = string();
assert(isstring(s) && isequal(size(s), [1 1]) && s == "");

% mat2str of logical values
assert(strcmp(mat2str(true), 'true'));
assert(strcmp(mat2str(false), 'false'));
assert(strcmp(mat2str([true false]), '[true false]'));
assert(strcmp(mat2str(logical([1; 0])), '[true;false]'));

% char input to num2str / int2str
assert(strcmp(num2str('abc'), 'abc'));
assert(strcmp(int2str('a'), '97'));

% regexprep and erase on cell and string arrays
assert(isequal(regexprep({'abc', 'bcd'}, 'b', 'X'), {'aXc', 'Xcd'}));
assert(isequal(regexprep(["abc", "xbx"], 'b', 'Y'), ["aYc", "xYx"]));
assert(isequal(erase({'abc', 'bcd'}, 'b'), {'ac', 'cd'}));

% logical sizes for strings / blanks
assert(isequal(strings(true), ""));
assert(isequal(blanks(true), ' '));

% sprintf's second output (errmsg) is empty on success
[s, e] = sprintf('%d', 7);
assert(strcmp(s, '7') && isempty(e));

% regexp's seventh default output is 'split'
[a, b, c, d, e, f, g] = regexp('ab', 'b');
assert(isequal(a, 2) && isequal(b, 2) && isequal(d, {'b'}));
assert(isequal(g, {'a', ''}));

% ── nargin(@f) ──────────────────────────────────────────────────────────
assert(nargin(@upper) == 1);
assert(nargin(@strrep) == 3);
assert(nargin(@strcmp) == 2);
assert(nargin(@strncmp) == 3);
assert(nargin(@num2str) == 2);
assert(nargin(@strjoin) == 2);
assert(nargin(@sscanf) == 3);
assert(nargin(@sprintf) == -1);
assert(nargin(@regexp) == -1);
assert(nargin(@newline) == 0);

disp('SUCCESS')

function check_err(stmt, id, msg)
  ok = false;
  try
    eval([stmt ';']);
  catch e
    ok = true;
    assert(strcmp(e.identifier, id), ...
      sprintf('%s: identifier ''%s'', expected ''%s''', stmt, e.identifier, id));
    assert(strcmp(e.message, msg), ...
      sprintf('%s: message ''%s'', expected ''%s''', stmt, e.message, msg));
  end
  assert(ok, sprintf('%s: expected an error', stmt));
end
