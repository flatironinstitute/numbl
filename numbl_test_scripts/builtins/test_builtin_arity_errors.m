% Arity errors, argument-type errors, and nargin(@f) for builtins: MATLAB's
% messages and identifiers.

% ── input count ─────────────────────────────────────────────────────────
check_err('isnan()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('isnan(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('isinf()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('isfinite(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('isreal()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('isreal(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('and(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('and(1, 2, 3)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('or(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('not()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('not(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('real()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('imag(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('conj()', 'MATLAB:minrhs', 'Not enough input arguments.');
% angle is an M-file in MATLAB, which uses different "too many" identifiers
check_err('angle()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('angle(1, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('[a, b] = angle(1)', 'MATLAB:TooManyOutputs', ...
  'Too many output arguments.');

% ── output count ────────────────────────────────────────────────────────
check_err('[a, b] = isnan(1)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('[a, b] = and(1, 0)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('[a, b] = real(1)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('x = disp(1)', 'MATLAB:maxlhs', 'Too many output arguments.');

% ── argument types ──────────────────────────────────────────────────────
check_err('isnan({})', 'MATLAB:math:mustBeNumericCharOrLogical', ...
  'Invalid data type. Argument must be numeric, char, or logical.');
check_err('real({})', 'MATLAB:math:mustBeNumericCharOrLogical', ...
  'Invalid data type. Argument must be numeric, char, or logical.');
check_err('and(1, {})', 'MATLAB:math:mustBeNumericCharOrLogical', ...
  'Invalid data type. Argument must be numeric, char, or logical.');
check_err('or(1i, 0)', 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err('or([1i 0], 0)', 'MATLAB:math:mustBeReal', 'Argument must be real.');
check_err('not(1i)', 'MATLAB:Not:operandsNotReal', 'Operands must be real.');

% char input is numeric (character codes)
assert(isequal(isnan('ab'), logical([0 0])));
assert(isequal(isinf('a'), false));
assert(isequal(isfinite('ab'), logical([1 1])));
assert(real('a') == 97);
assert(imag('a') == 0);
assert(conj('a') == 97);
assert(angle('a') == 0);
assert(isequal(not('a'), false));
assert(isequal(and(1, 'a'), true));

% sparse input to the predicates
assert(isequal(full(isnan(sparse([1 NaN]))), logical([0 1])));

% isreal on non-numeric containers
assert(~isreal(struct('a', {1, 2})));
assert(~isreal(["a", "b"]));
assert(isreal('abc'));

% ── nargin(@f) ──────────────────────────────────────────────────────────
assert(nargin(@isnan) == 1);
assert(nargin(@isinf) == 1);
assert(nargin(@isfinite) == 1);
assert(nargin(@isreal) == 1);
assert(nargin(@and) == 2);
assert(nargin(@or) == 2);
assert(nargin(@not) == 1);
assert(nargin(@real) == 1);
assert(nargin(@imag) == 1);
assert(nargin(@conj) == 1);
assert(nargin(@angle) == 1);

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
