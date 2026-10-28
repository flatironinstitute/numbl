% Builtins that work on the calling scope (eval, evalin, assignin, exist,
% which, nargin, nargout, narginchk, nargoutchk, inputname, str2num, ...):
% arity errors, and the same behavior when called directly, through a
% function handle, through feval, and through builtin().

% ── arity ───────────────────────────────────────────────────────────────
check_err('eval()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('eval(''1'', ''2'', ''3'')', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('evalin(''base'')', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('evalin(''base'', ''1'', ''2'', ''3'')', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('assignin(''base'', ''x'')', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('assignin(''base'', ''x'', 1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('q = assignin(''base'', ''x'', 1)', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('exist()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('exist(''x'', ''var'', 3)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('exist(''x'', ''bogus'')', 'MATLAB:exist:secondOptionalInputString', ...
  'The optional second input to exist must be ''var'', ''builtin'', ''class'', ''dir'' or ''file''.');
check_err('which()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('inputname()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('run()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('run(''a'', ''b'')', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('q = run(''zz'')', 'MATLAB:TooManyOutputs', 'Too many output arguments.');
check_err('nargin(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('narginchk(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('nargoutchk(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('isfolder()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('isfile()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('isfolder(''a'', ''b'')', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('isa(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('superclasses()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('properties()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('q = clear(''zz'')', 'MATLAB:maxlhs', 'Too many output arguments.');
check_err('feval()', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('str2num()', 'MATLAB:minrhs', 'Not enough input arguments.');

% ── exist / which, called several ways ──────────────────────────────────
zz = 3;
assert(exist('zz') == 1);
assert(exist('zz', 'var') == 1);
assert(exist('zz_nope', 'var') == 0);
assert(exist('sin') == 5);
h = @exist;
assert(h('zz') == 1);
assert(feval('exist', 'zz') == 1);
assert(feval(@exist, 'zz', 'var') == 1);
assert(builtin('exist', 'zz') == 1);
assert(strcmp(which('zz'), 'variable'));
assert(startsWith(which('sin'), 'built-in'));
assert(isempty(which('zz_no_such_function')));
assert(probe_exist() == 0); % a function's own workspace

% ── evalin / assignin ───────────────────────────────────────────────────
set_in_caller('yy', 42);
assert(yy == 42);
feval(@set_in_caller, 'yy', 43);
assert(yy == 43);
assert(read_from_caller('yy') == 43);
assert(feval('read_from_caller', 'yy') == 43);
assert(evalin('base', 'yy + 1') == 44);
r = eval('yy * 2');
assert(r == 86);
eval('ww = 5;');
assert(ww == 5);
eval('error(''a:b'', ''boom'')', 'ww = 6;');
assert(ww == 6);

% ── nargin / nargout / narginchk / nargoutchk / inputname ───────────────
assert(nargin('two_in_one_out') == 2);
assert(nargin(@two_in_one_out) == 2);
assert(nargout('two_in_one_out') == 1);
assert(nargout(@set_in_caller) == 0);
assert(nargin('varargin_fn') == -2);
[n, m] = counts(1, 2);
assert(n == 2 && m == 2);
check_err('needs_two(1)', 'MATLAB:narginchk:notEnoughInputs', 'Not enough input arguments.');
check_err('needs_two(1, 2, 3, 4)', 'MATLAB:narginchk:tooManyInputs', 'Too many input arguments.');
needs_two(1, 2);
check_err('needs_an_output()', 'MATLAB:nargoutchk:notEnoughOutputs', 'Not enough output arguments.');
q = needs_an_output(); %#ok<NASGU>
assert(strcmp(name_of_arg(zz), 'zz'));
assert(strcmp(name_of_arg(zz + 1), ''));

% ── str2num ─────────────────────────────────────────────────────────────
[x, ok] = str2num('[1 2; 3 4]'); %#ok<ST2NM>
assert(isequal(x, [1 2; 3 4]) && islogical(ok) && ok);
[x, ok] = str2num('1 +'); %#ok<ST2NM>
assert(isempty(x) && ~ok);
assert(str2num('42') == 42); %#ok<ST2NM>

disp('SUCCESS');

function r = probe_exist()
  r = exist('zz', 'var');
end

function set_in_caller(name, v)
  assignin('caller', name, v);
end

function v = read_from_caller(name)
  v = evalin('caller', name);
end

function y = two_in_one_out(a, b)
  y = a + b;
end

function varargin_fn(a, varargin) %#ok<INUSD>
end

function [n, m] = counts(varargin)
  n = nargin;
  m = nargout;
end

function needs_two(varargin)
  narginchk(2, 3);
end

function varargout = needs_an_output()
  nargoutchk(1, 1);
  varargout{1} = 1;
end

function s = name_of_arg(v) %#ok<INUSD>
  s = inputname(1);
end

function check_err(stmt, id, msg)
  ok = false;
  try
    evalin('caller', [stmt ';']);
  catch e
    ok = true;
    assert(strcmp(e.identifier, id), ...
      sprintf('%s: identifier ''%s'', expected ''%s''', stmt, e.identifier, id));
    assert(strcmp(e.message, msg), ...
      sprintf('%s: message ''%s'', expected ''%s''', stmt, e.message, msg));
  end
  assert(ok, sprintf('%s: expected an error', stmt));
end
