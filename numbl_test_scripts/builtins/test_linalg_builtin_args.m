% Linear algebra, sparse, introspection and misc builtins: arity and type
% errors with MATLAB's identifiers, nargin/nargout of handles, and results
% that used to be wrong.

% Complex SVD and QZ need the native LAPACK addon in numbl.
has_lapack = true;
try
  svd(1+2i);
catch e
  if isempty(strfind(e.message, 'requires LAPACK')), rethrow(e); end
  has_lapack = false;
end

% ── input and output counts ─────────────────────────────────────────────
check_err('norm()', 'MATLAB:minrhs');
check_err('norm(1, 2, 3)', 'MATLAB:maxrhs');
check_err('[a, b] = norm(1)', 'MATLAB:maxlhs');
check_err('[a, b] = det(1)', 'MATLAB:maxlhs');
check_err('[a, b] = isempty(1)', 'MATLAB:maxlhs');
check_err('length(1, 2)', 'MATLAB:maxrhs');
check_err('ndims()', 'MATLAB:minrhs');
check_err('qrupdate(eye(2), eye(2), [1;1])', 'MATLAB:minrhs');
check_err('plus(1)', 'MATLAB:minrhs');
check_err('plus(1, 2, 3)', 'MATLAB:maxrhs');
check_err('uminus(1, 2)', 'MATLAB:maxrhs');
check_err('clc(1)', 'MATLAB:maxrhs');
check_err('filesep(1)', 'MATLAB:maxrhs');
check_err('isvarname(''a'', ''b'')', 'MATLAB:maxrhs');
check_err('getappdata(0, ''a'', 3)', 'MATLAB:maxrhs');
check_err('setappdata(0, ''a'')', 'MATLAB:minrhs');
check_err('sparse()', 'MATLAB:minrhs');
check_err('sparse([1 2], [1 2], [3 4], 3)', 'MATLAB:minrhs');
check_err('sparse(1, 2, 3, 4, 5, 6, 7)', 'MATLAB:maxrhs');
check_err('spalloc(2, 3)', 'MATLAB:minrhs');
check_err('amd(sparse(eye(3)), 1, 2)', 'MATLAB:maxrhs');
% M-files use different "too many" identifiers
check_err('[a, b] = trace(1)', 'MATLAB:TooManyOutputs');
check_err('kron(1, 2, 3)', 'MATLAB:TooManyInputs');
check_err('[a, b] = fileparts(''a/b.c'', 1)', 'MATLAB:TooManyInputs');
check_err('now(1)', 'MATLAB:TooManyInputs');
check_err('usejava(''jvm'', 2)', 'MATLAB:TooManyInputs');
check_err('sprand(3, 3, 0.5, 0.1, 1)', 'MATLAB:TooManyInputs');
check_err('spalloc(2, 3, 4, 5, 6)', 'MATLAB:TooManyInputs');
check_err('fullfile()', 'MATLAB:narginchk:notEnoughInputs');

% ── nargin / nargout ────────────────────────────────────────────────────
assert(nargin(@norm) == 2);
assert(nargin(@svd) == 3);
assert(nargin(@kron) == 2);
assert(nargin(@size) == -1);
assert(nargin(@blkdiag) == -1);
assert(nargin(@(varargin) 1) == -1);
assert(nargin(@(x, varargin) 1) == -2);
assert(nargout(@svd) == 3);
assert(nargout(@lu) == 5);
assert(nargout(@plus) == 1);
assert(nargout(@size) == -1);
assert(nargout('svd') == 3);
assert(nargout("norm") == 1);
assert(nargout(@(x) x) == -1);
assert(nargout(@la_two_outputs) == 2);
assert(nargout('la_two_outputs') == 2);
assert(nargout(@la_varargout) == -2);
check_err('nargin(5)', 'MATLAB:narginout:BadInput');
check_err('nargout(5)', 'MATLAB:narginout:BadInput');
check_err('nargin(''nosuchfn_xyz'')', 'MATLAB:narginout:notValidMfile');
check_err('nargout(''nosuchfn_xyz'')', 'MATLAB:narginout:notValidMfile');

% ── argument types ──────────────────────────────────────────────────────
check_err('norm(''abc'')', 'MATLAB:norm:inputType');
check_err('norm({1})', 'MATLAB:norm:inputType');
check_err('vecnorm(''ab'')', 'MATLAB:vecnorm:inputType');
check_err('det(''a'')', 'MATLAB:math:mustBeFloat');
check_err('inv(''a'')', 'MATLAB:inv:inputType');
check_err('svd(sparse([1 2;3 4]))', 'MATLAB:svd:noSVDsparse');
check_err('rank(sparse([1 2;3 4]))', 'MATLAB:svd:noSVDsparse');
check_err('pinv(sparse(2))', 'MATLAB:svd:noSVDsparse');
check_err('null(sparse([1 1]))', 'MATLAB:svd:noSVDsparse');
check_err('linsolve(sparse(2), 2)', 'MATLAB:linsolve:sparseNotSupported');
check_err('pagemtimes(sparse(2), 2)', 'MATLAB:pagemtimes:sparseNotSupported');
check_err('pagemtimes(2, "none", 3)', ...
  'MATLAB:pagemtimes:threeInputNotSupported');
check_err('qz(sparse([1 2;3 4]), [5 6;7 8])', 'MATLAB:qz:noQZsparse');
check_err('[V, D] = eig(sparse([2 0;0 3]))', ...
  'MATLAB:eig:useEigsForSparseStandardVector');
check_err('istriu(''ab'')', 'MATLAB:math:mustBeFloatOrLogical');
check_err('bandwidth({1})', 'MATLAB:bandwidth:inputType');
check_err('isbanded([1 2], 1.5, 0)', 'MATLAB:isbanded:nonNegIntScalar');
check_err('isbanded([1 2], -1, 0)', 'MATLAB:isbanded:nonNegIntScalar');
check_err('trace({1})', 'MATLAB:diag:inputType');
check_err('dot({1}, {1})', 'MATLAB:math:mustBeNumericCharOrLogical');
check_err('symrcm(''abc'')', 'MATLAB:math:mustBeFloatOrLogical');
check_err('sparse(''ab'')', 'MATLAB:sparse:invalidMatrix');
check_err('sparse(1+1i, 1, 2)', 'MATLAB:sparsfcn:nonPositiveIndicesOfSparse');
check_err('speye(''a'')', 'MATLAB:speye:invalidTypename');
check_err('fieldnames(5)', 'MATLAB:fieldnames:InvalidInput');
check_err('fileparts(5)', 'MATLAB:fileparts:MustBeChar');
check_err('jsondecode(5)', 'MATLAB:json:InvalidInput');
check_err('sprand(3, 3)', 'MATLAB:sprand:TwoInputs');

% ── option and dimension arguments ──────────────────────────────────────
check_err('norm([1 2;3 4], 3)', 'MATLAB:norm:unknownNorm');
check_err('norm([1 2], 1+1i)', 'MATLAB:norm:unknownNorm');
check_err('vecnorm([3 4], 2, 1.5)', ...
  'MATLAB:getdimarg:dimensionMustBePositiveInteger');
check_err('dot([1 2], [3 4], 0)', 'MATLAB:getdimarg:invalidDim');
check_err('cross([1 2 3], [4 5 6], 1.5)', 'MATLAB:size:invalidDim');
check_err('svd([1 2;3 4], "foo")', 'MATLAB:svd:invalidOption');
check_err('svd([1 2;3 4], 1)', 'MATLAB:svd:invalidOption');
check_err('qr([1 2;3 4], "foo")', 'MATLAB:qr:unknownOption');
check_err('qr([1 2;3 4], 1)', 'MATLAB:qr:notSupportFill');
check_err('lu(rand(3), 0.5)', 'MATLAB:factorFlags:invalidPermFlag');
check_err('[L, U, P, Q] = lu(rand(3))', 'MATLAB:lu:DenseThreeOutputOnly');
check_err('[a, b, c] = chol([4 0;0 9])', 'MATLAB:chol:DenseTwoOutputOnly');
check_err('chol([1 2;2 1])', 'MATLAB:posdef');
check_err('chol([2+1i 0; 0 2])', ...
  'MATLAB:chol:matrixMustBePosDefWithRealDiag');
check_err('null([1 1], 1+1i)', 'MATLAB:null:invalidSecondArgument');
check_err('rank([1 2;2 4], 1+1i)', 'MATLAB:rank:invalidSecondArgument');
check_err('pinv([1 2;2 4], [1 2])', 'MATLAB:pinv:invalidSecondArgument');
check_err('size(zeros(2,3), 0)', 'MATLAB:size:invalidDim');
check_err('size(zeros(2,3), 1.5)', 'MATLAB:size:invalidDim');
check_err('[a, b] = size(1, 1)', 'MATLAB:size:NumOutNotEqualNumDims');
check_err('x = spparms(''bandden'', 0)', 'MATLAB:spparms:TooManyOutputs');

% ── square-matrix checks ────────────────────────────────────────────────
check_err('det([1 2 3])', 'MATLAB:square');
check_err('inv([1 2 3])', 'MATLAB:square');
check_err('chol([1 2 3])', 'MATLAB:square');
check_err('trace([1 2 3; 4 5 6])', 'MATLAB:trace:square');
check_err('expm([1 2 3])', 'MATLAB:expm:inputMustBeSquare');
check_err('eig([1 2 3])', 'MATLAB:eig:inputMustBeSquareStandard');
check_err('rcond([1 2 3])', 'MATLAB:rcond:notSquareMatrix');
check_err('linsolve([1 2;3 4], [1;2;3])', 'MATLAB:dimagree');

% ── complex scalars ─────────────────────────────────────────────────────
assert(det(1+2i) == 1+2i);
if has_lapack
  assert(abs(svd(1+2i) - sqrt(5)) < 1e-12);
end
assert(eig(1+2i) == 1+2i);
assert(lu(1+2i) == 1+2i);
assert(abs(linsolve(1+1i, 2) - (1-1i)) < 1e-12);
assert(dot(1+2i, 3) == 3-6i);
assert(abs(expm(1+2i) - exp(1+2i)) < 1e-12);
assert(pagetranspose(1+2i) == 1+2i);
[q, r] = qr(1+2i);
assert(abs(q*r - (1+2i)) < 1e-12);

% ── sparse input ────────────────────────────────────────────────────────
assert(det(sparse([1 2;3 4])) == -2);
assert(trace(sparse([1 2;3 4])) == 5);
assert(isequal(eig(sparse([2 0;0 3])), [2;3]));
x = vecnorm(sparse([3 4]));
assert(issparse(x) && full(x) == 5);
x = dot(sparse([1 2]), sparse([3 4]));
assert(issparse(x) && full(x) == 11);
x = dot(sparse([1 2]), [3 4]);
assert(~issparse(x) && x == 11);
x = cross(sparse([1 0 0]), sparse([0 1 0]));
assert(issparse(x) && isequal(full(x), [0 0 1]));
x = blkdiag(sparse(1), 2);
assert(issparse(x) && isequal(full(x), [1 0;0 2]));
x = spdiags([1 2 3]', 0, eye(3));
assert(issparse(x) && isequal(full(x), diag([1 2 3])));
x = spdiags([1 2 3]', 0, sparse(eye(3)));
assert(issparse(x) && isequal(full(x), diag([1 2 3])));
x = pagetranspose(sparse([1 2]));
assert(issparse(x) && isequal(full(x), [1;2]));
x = kron(sparse(2), true);
assert(issparse(x) && full(x) == 2);
x = spconvert([1 1 2 3; 2 2 4 5]);
assert(isequal(full(x), [2+3i 0; 0 4+5i]));

% ── results that used to be wrong ───────────────────────────────────────
assert(lu(5) == 5);
[q, r] = qr(-5);
assert(q == 1 && r == -5);
[U, S] = svd([1 2;3 4]);
assert(isequal(size(S), [2 2]) && S(2,1) == 0);
assert(norm(U*S*S'*U' - [1 2;3 4]*[1 2;3 4]') < 1e-10);
[U, S, V] = svd([1 2;3 4], "vector");
assert(isequal(size(S), [2 1]));
S = svd([1 2;3 4], "matrix");
assert(isequal(size(S), [2 2]) && S(1,2) == 0);
[U, S, V] = svd([1 2;3 4;5 6], "econ", "vector");
assert(isequal(size(S), [2 1]) && isequal(size(U), [3 2]));
A = [1 2;3 4;5 6];
[Q, R, E] = qr(A, 0);
assert(isequal(E, [2 1]));
[Q, R, E] = qr(A, "econ");
assert(isequal(E, [0 1;1 0]) && isequal(size(Q), [3 2]));
[Q, R, E] = qr(A, "vector");
assert(isequal(E, [2 1]) && isequal(size(Q), [3 3]));
[Q, R, E] = qr(A, "econ", "vector");
assert(isequal(E, [2 1]) && isequal(size(Q), [3 2]));
[a, b] = size(zeros(2,3,4));
assert(a == 2 && b == 12);
[a, b, c, d] = size(zeros(2,3,4));
assert(isequal([a b c d], [2 3 4 1]));
assert(isequal(size(zeros(2,3,4), [1 3]), [2 4]));
assert(isequal(size(zeros(2,3,4), 1, 3), [2 4]));
[a, b] = size(zeros(2,3,4), [1 3]);
assert(a == 2 && b == 4);
assert(size(zeros(2,3), true) == 2);
assert(size(zeros(2,3), 5) == 1);
assert(abs(cond([1 2;3 4], 'fro') - 15) < 1e-10);
assert(abs(cond([1 2;3 4], "inf") - 21) < 1e-10);
assert(norm([1 2], "inf") == 2);
[p1, p2, p3] = fileparts("x/y.m");
assert(isstring(p1) && isstring(p2) && isstring(p3));
assert(p2 == "y" && p3 == ".m");
[p1, p2] = fileparts('x/y.m');
assert(ischar(p1) && strcmp(p2, 'y'));
assert(odeget(struct('A', []), 'A', 5) == 5);
assert(odeget(struct('A', 1), 'A', 5) == 1);
assert(isequal(fieldnames(struct('a', 1), '-full'), {'a'}));
[m, id] = lastwarn;
assert(ischar(id));
spparms('bandden', 0);
spparms('default');
if has_lapack
  x = qz(2, 3);
  assert(x == 2);
  AA = qz([1 2;3 4], [5 6;7 8]);
  assert(isequal(size(AA), [2 2]));
end
[X, Y, Z] = peaks(3);
assert(isequal(X, [-3 0 3; -3 0 3; -3 0 3]));
assert(isequal(Y, X'));
Zf = 3*(1-X).^2.*exp(-(X.^2) - (Y+1).^2) ...
  - 10*(X/5 - X.^3 - Y.^5).*exp(-X.^2-Y.^2) ...
  - 1/3*exp(-(X+1).^2 - Y.^2);
assert(max(abs(Z(:) - Zf(:))) < 1e-12);
Z1 = peaks(3);
assert(isequal(Z1, Z));

% ── numeric-like char input to M-file functions ─────────────────────────
assert(kron('a', 2) == 194);
assert(dot('ab', 'cd') == 19403);
assert(isempty(blkdiag()));
assert(pagetranspose(3) == 3);
assert(islogical(pagetranspose(true)));

disp('SUCCESS')

function check_err(stmt, id)
  ok = false;
  try
    eval([stmt ';']);
  catch e
    ok = true;
    assert(strcmp(e.identifier, id), ...
      sprintf('%s: identifier ''%s'', expected ''%s''', stmt, e.identifier, id));
  end
  assert(ok, sprintf('%s: expected an error', stmt));
end

function [a, b] = la_two_outputs(x)
  a = x;
  b = x;
end

function [a, varargout] = la_varargout(x)
  a = x;
  varargout = {};
end
