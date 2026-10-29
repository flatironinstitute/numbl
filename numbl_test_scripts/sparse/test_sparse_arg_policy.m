% Sparse arguments to builtins: which keep sparsity, which return full
% results, and which reject sparse input (MATLAB's errors). Verified in
% MATLAB.

S = sparse([4 0 1; 0 3 0; 1 0 5]);
V = sparse([1; 0; 2; 0]);
One = sparse(7);
F = full(S);

% Builtins that take sparse natively keep it.
assert(issparse(S'));
assert(issparse(-S));
assert(issparse(S + S));
assert(issparse(S .* 2));
assert(issparse(sum(S)));
assert(issparse(reshape(S, 1, 9)));
assert(issparse(tril(S)));
assert(issparse(kron(S, S)));
assert(issparse(fliplr(S)));
assert(isequal(find(S), find(F)));
assert(nnz(S) == 5);
assert(isequal(nonzeros(S), [4; 1; 3; 1; 5]));
assert(isequal(size(S), [3 3]));
assert(numel(S) == 9);
assert(isa(S, 'double') && isnumeric(S) && isfloat(S) && ~isinteger(S));
assert(isreal(S) && ismatrix(S) && ~isempty(S) && ~isscalar(S));
assert(strcmp(class(S), 'double'));
st = struct('a', S);
assert(issparse(st.a));
st2 = setfield(struct(), 'b', S);
assert(issparse(st2.b));
[d1, d2] = deal(S);
assert(issparse(d1) && issparse(d2));
assert(issparse(feval(@(x) x, S)));
assert(isequal(cellfun(@issparse, {S, F}), [true false]));

% Builtins without sparse support compute on the full matrix and return
% full results, as MATLAB does.
[X, Y] = meshgrid(V);
assert(~issparse(X) && isequal(X, meshgrid(full(V))));
assert(~issparse(Y));
assert(isequal(magic(One), magic(7)) && ~issparse(magic(One)));
assert(isequal(poly(S), poly(F)) && ~issparse(poly(S)));
assert(strcmp(num2str(S), num2str(F)));
assert(strcmp(int2str(S), int2str(F)));
assert(isequal(isspace(S), false(3)) && ~issparse(isspace(S)));
assert(isequal(colon(One, 9), 7:9) && ~issparse(colon(One, 9)));
assert(isequal(peaks(One), peaks(7)));
assert(isequal(groupcounts(S), groupcounts(F)));
tf = ismember(S, One);
assert(isequal(full(tf), false(3)));
z = complex(S, S);
assert(isequal(real(z), F) && isequal(imag(z), F));
[th, r] = cart2pol(S, S);
assert(max(abs(full(r(:)) - sqrt(2) * F(:))) < 1e-12);
assert(isequal(hours(One), hours(7)));
assert(strcmp(dec2bin(One), '111'));
assert(isequal(eps(S), eps(F)));
assert(isequal(floor(full(conj(S))), F));
assert(max(abs(polyval([1 2 3], V) - polyval([1 2 3], full(V)))) == 0);

% Builtins that MATLAB defines only for full input reject sparse.
assert(strcmp(errId(@() int8(S)), 'MATLAB:unimplementedSparseType'));
assert(strcmp(errId(@() uint16(One)), 'MATLAB:unimplementedSparseType'));
assert(strcmp(errId(@() char(One)), 'MATLAB:unimplementedSparseType'));
assert(strcmp(errId(@() bitand(S, 1)), 'MATLAB:Bitand:sparseArgs'));
assert(strcmp(errId(@() bitand(1, S)), 'MATLAB:Bitand:sparseArgs'));
assert(strcmp(errId(@() bitor(S, 1)), 'MATLAB:Bitor:sparseArgs'));
assert(strcmp(errId(@() bitxor(S, 1)), 'MATLAB:bitxor:sparseArgs'));
assert(strcmp(errId(@() bitshift(1, One)), 'MATLAB:bitshift:sparseArgs'));
assert(strcmp(errId(@() erf(S)), 'MATLAB:erf:notFullReal'));
assert(strcmp(errId(@() gamma(S)), 'MATLAB:gamma:notFullReal'));
assert(strcmp(errId(@() gammaln(S)), 'MATLAB:gammaln:notFullReal'));
assert(strcmp(errId(@() svd(S)), 'MATLAB:svd:noSVDsparse'));
assert(strcmp(errId(@() pinv(One)), 'MATLAB:svd:noSVDsparse'));
assert(strcmp(errId(@() rank(S)), 'MATLAB:svd:noSVDsparse'));
assert(strcmp(errId(@() null(S)), 'MATLAB:svd:noSVDsparse'));
assert(strcmp(errId(@() linsolve(S, [1; 2; 3])), 'MATLAB:linsolve:sparseNotSupported'));
assert(strcmp(errId(@() pagemtimes(eye(2), One)), 'MATLAB:pagemtimes:sparseNotSupported'));
assert(strcmp(errId(@() jsonencode(S)), 'MATLAB:json:UnsupportedSparseDataType'));
assert(strcmp(errId(@() typecast(One, 'uint8')), 'MATLAB:typecast:mustBeFull'));
assert(strcmp(errId(@() sprintf('%d', One)), 'MATLAB:printf:invalidInputAttribute'));
assert(strcmp(errId(@() fft(S)), 'MATLAB:fftfcn:SparseInputType'));
% ... but a sparse length argument is just a number.
assert(isequal(fft(1, One), fft(1, 7)));

% The arity check comes first.
assert(strcmp(errId(@() erf(S, 1)), 'MATLAB:TooManyInputs') || ...
       strcmp(errId(@() erf(S, 1)), 'MATLAB:maxrhs'));

disp('SUCCESS');

function id = errId(f)
  id = '';
  try
    f();
  catch e
    id = e.identifier;
  end
end
