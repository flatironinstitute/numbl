% Arity and argument-type errors for utility, validation, dictionary,
% duration, cell/struct, and system builtins (MATLAB's messages and
% identifiers).

% ── utility ─────────────────────────────────────────────────────────────
check_err('isequal(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('[a, b] = isequal(1, 1)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('x = assert(true)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('x = error(''a'')', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('rethrow()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('MException(''a:b'')', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('MException(1, ''x'')', 'MATLAB:dispatcher:noMatchingConstructor', ...
  'No constructor ''MException'' with matching signature found.');
check_err('MException(''a:b'', 1)', 'MATLAB:MException:MessageMustBeChar', ...
  'MESSAGE input must be a character vector or string scalar.');

% ── validation (M-files with no outputs) ────────────────────────────────
check_err('mustBePositive()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('mustBePositive(1, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('x = mustBePositive(1)', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('mustBeMember(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('mustBeInRange(1, 0)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('validateattributes(1, {''numeric''})', 'MATLAB:minrhs', ...
  'Not enough input arguments.');
check_err('x = validateattributes(1, {''numeric''}, {})', 'MATLAB:maxlhs', ...
  'Too many output arguments.');
mustBeInRange(3, 0, 5);
mustBeVector([], 'allow-all-empties');
% a lower bound of 0 is honored
ok = false;
try
  mustBeInRange(-1, 0, 5);
catch
  ok = true;
end
assert(ok);

% ── dictionary ──────────────────────────────────────────────────────────
d = dictionary("a", 1, "b", 2);
check_err('dictionary(1)', 'MATLAB:dictionary:IncorrectNumberOfInputs', ...
  'Incorrect number of input arguments. Each key must be followed by a value.');
check_err('dictionary(1, 2, 3)', 'MATLAB:dictionary:IncorrectNumberOfInputs', ...
  'Incorrect number of input arguments. Each key must be followed by a value.');
check_err('configureDictionary("string")', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('configureDictionary(1, 2)', 'MATLAB:string:MustBeStringScalarOrCharacterVector', ...
  'Argument must be a text scalar.');
check_err('keys(5)', 'MATLAB:UndefinedFunction', ...
  'Undefined function ''keys'' for input arguments of type ''double''.');
check_err('numEntries(5)', 'MATLAB:UndefinedFunction', ...
  'Undefined function ''numEntries'' for input arguments of type ''double''.');
check_err('isKey(d)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('isKey(d, 1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('numEntries(d, 1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('remove(d)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('remove(d, 1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('insert(d, "c")', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('lookup(d, "a", 1)', 'MATLAB:dictionary:OddNumberNVPairs', ...
  'Incorrect number of arguments. Each parameter name must be followed by a corresponding value.');
check_err('[a, b] = keys(d)', 'MATLAB:maxlhs', 'Too many output arguments.');
assert(lookup(d, "z", "FallbackValue", 7) == 7);
[kt, vt] = types(d);
assert(kt == "string" && vt == "double");

% ── durations (M-files) ─────────────────────────────────────────────────
check_err('seconds(1, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('seconds(''a'')', 'MATLAB:duration:InvalidDurationData', ...
  'Input data must be a real, numeric array, or a duration array.');
assert(seconds(minutes(2)) == 120);

% ── cell / struct ───────────────────────────────────────────────────────
check_err('cell2mat({1}, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('[a, b] = cell2mat({1})', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('cell2struct({1})', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('cell2struct({1}, {''a''}, 1, 1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('struct2cell(struct(''a'', 1), 1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('rmfield(struct(''a'', 1))', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('rmfield(struct(''a'', 1), ''a'', 1)', 'MATLAB:TooManyInputs', ...
  'Too many input arguments.');
check_err('num2cell(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');

% ── system ──────────────────────────────────────────────────────────────
check_err('tic(1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('clock(1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('ismac(1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('etime(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('etime(1, 2, 3)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('computer(''x'')', 'MATLAB:badopt', 'Unknown command option.');
check_err('[a, b, c, d] = computer', 'MATLAB:maxlhs', 'Too many output arguments.');
assert(ischar(version('-release')));

% ── nargin(@f) ──────────────────────────────────────────────────────────
assert(nargin(@isequal) == -1);
assert(nargin(@assert) == -1);
assert(nargin(@rethrow) == 1);
assert(nargin(@mustBePositive) == 1);
assert(nargin(@mustBeMember) == 2);
assert(nargin(@configureDictionary) == 2);
assert(nargin(@cell2mat) == 1);
assert(nargin(@cell2struct) == 3);
assert(nargin(@rmfield) == 2);
assert(nargin(@num2cell) == 2);
assert(nargin(@seconds) == 1);
assert(nargin(@etime) == 2);
assert(nargin(@tic) == 0);

disp('SUCCESS')

function check_err(stmt, id, msg)
  ok = false;
  d = dictionary("a", 1, "b", 2); %#ok<NASGU> used by eval'd statements
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
