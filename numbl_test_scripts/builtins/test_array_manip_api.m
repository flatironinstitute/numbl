% Array manipulation builtins: arity errors, scalar/logical/char/complex
% inputs, and circshift with a dimension argument.

% ── Arity errors ───────────────────────────────────────────────────────
check_err('transpose(1, 2)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('fliplr(1, 2)', 'MATLAB:TooManyInputs', 'Too many input arguments.');
check_err('permute(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('repmat(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('circshift(1, 1, 1, 1)', 'MATLAB:maxrhs', 'Too many input arguments.');
check_err('reshape(1)', 'MATLAB:minrhs', 'Not enough input arguments.');
check_err('ndgrid()', 'MATLAB:ndgrid:NotEnoughInputs', 'Not enough input arguments.');
check_err('meshgrid()', 'MATLAB:meshgrid:NotEnoughInputs', 'Not enough input arguments.');
try
    [a, b, c, d] = meshgrid(1:2);
    error('expected error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:TooManyOutputs'));
end
try
    [a, b] = reshape(1, 1, 1);
    error('expected error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:maxlhs'));
end
assert(nargin(@fliplr) == 1);
assert(nargin(@diag) == 2);
assert(nargin(@circshift) == 3);
assert(nargin(@meshgrid) == 3);
assert(nargin(@ind2sub) == 2);
assert(nargin(@reshape) == -1);

% ── Concatenation ──────────────────────────────────────────────────────
x = horzcat();
assert(isequal(size(x), [0 0]));
x = vertcat();
assert(isequal(size(x), [0 0]));
x = cat(3);
assert(isequal(size(x), [0 0]));
x = cat(3, 1i, 2);
assert(isequal(size(x), [1 1 2]));
assert(x(1) == 1i && x(2) == 2);
x = cat(3, true, false);
assert(islogical(x) && isequal(size(x), [1 1 2]));
check_err('cat(1+1i, 1, 2)', 'MATLAB:catenate:invalidDimension', ...
    'Dimension argument must be a real, positive, integer scalar.');

% ── diag ───────────────────────────────────────────────────────────────
x = diag(true);
assert(islogical(x) && x);
x = diag(logical([1 0 1]));
assert(islogical(x) && isequal(x, logical(eye(3) .* [1 0 1])));
check_err('diag([1 2], 1+1i)', 'MATLAB:diag:kthDiagInputNotInteger', ...
    'Kth diagonal input must be an integer scalar.');

% ── flips and rotations ────────────────────────────────────────────────
assert(fliplr(1+2i) == 1+2i);
assert(islogical(flipud(true)));
assert(islogical(flip(true)));
assert(flip(1+2i) == 1+2i);
assert(rot90(1+1i) == 1+1i);
assert(islogical(rot90(true)));
assert(strcmp(flip('abc'), 'cba'));
m = ['ab'; 'cd'];
assert(isequal(fliplr(m), ['ba'; 'dc']));
assert(isequal(flipud(m), ['cd'; 'ab']));
assert(isequal(flip(m), ['cd'; 'ab']));
assert(isequal(flip(m, 2), ['ba'; 'dc']));
x = fliplr(logical([1 0 0]));
assert(islogical(x) && isequal(x, logical([0 0 1])));

% ── repmat / repelem ───────────────────────────────────────────────────
x = repmat('ab', 3, 1);
assert(ischar(x) && isequal(x, ['ab'; 'ab'; 'ab']));
x = repmat('ab', 2);
assert(isequal(x, ['abab'; 'abab']));
x = repmat('xy', 1, 3);
assert(strcmp(x, 'xyxyxy'));
x = repmat(1+1i, 2);
assert(isequal(x, (1+1i) * ones(2)));
check_err('repmat(1, 1+1i)', 'MATLAB:repmat:complexReplications', ...
    'Complex replication factors are not supported.');
x = repelem(1+1i, 2);
assert(isequal(x, [1+1i 1+1i]));
x = repelem(true, 2);
assert(islogical(x) && isequal(x, [true true]));
x = repelem(logical([1 0]), 2);
assert(islogical(x) && isequal(x, logical([1 1 0 0])));
x = repelem(logical([1 0]), 2, 1);
assert(islogical(x) && isequal(x, logical([1 0; 1 0])));
check_err('repelem([1 2], 1+1i)', 'MATLAB:repelem:complexReplications', ...
    'Complex replication factors are not supported.');

% ── circshift ──────────────────────────────────────────────────────────
A = [1 2; 3 4];
assert(isequal(circshift(A, 1, 2), [2 1; 4 3]));
assert(isequal(circshift(A, 1, 1), [3 4; 1 2]));
assert(isequal(circshift(A, 1), [3 4; 1 2]));
assert(isequal(circshift(A, [1 1]), [4 3; 2 1]));
assert(isequal(circshift(1:5, 2), [4 5 1 2 3]));
assert(isequal(circshift(1:5, -1, 2), [2 3 4 5 1]));
assert(isequal(circshift(1:5, 1, 1), 1:5));
B = reshape(1:3, 1, 1, 3);
assert(isequal(circshift(B, 1), reshape([3 1 2], 1, 1, 3)));
assert(isequal(circshift(1:3, 1, 3), 1:3));
check_err('circshift(1:3, 1i)', 'MATLAB:circshift:InvalidShiftType', ...
    'Invalid shift type: must be a real finite integer vector.');
check_err('circshift(1:3, [1 2], 1)', 'MATLAB:circshift:NonScalarDim', ...
    'In CIRCSHIFT(X,K,DIM), K and DIM must be scalars.');
check_err('circshift(1:3, 1, 0)', ...
    'MATLAB:getdimarg:dimensionMustBePositiveInteger', ...
    'Dimension argument must be a positive integer scalar within indexing range.');
assert(strcmp(circshift('abc', 1), 'cab'));
assert(isequal(circshift({1, 2, 3}, 1), {3, 1, 2}));
assert(circshift(1+1i, 1) == 1+1i);
assert(islogical(circshift(true, 1)));
x = circshift(logical([1 0 0]), 1);
assert(islogical(x) && isequal(x, logical([0 1 0])));
x = circshift([1+1i 2 3], 1);
assert(isequal(x, [3 1+1i 2]));

% ── meshgrid / ndgrid ──────────────────────────────────────────────────
[X, Y] = meshgrid([1 2] + 1i, [3 4]);
assert(isequal(X, [1+1i 2+1i; 1+1i 2+1i]));
assert(isequal(Y, [3 3; 4 4]));
x = meshgrid(true);
assert(islogical(x) && x);
x = ndgrid([1 2] + 1i);
assert(isequal(x, [1+1i; 2+1i]));
[a, b] = ndgrid(1:2, 1:3);
assert(isequal(a, [1 1 1; 2 2 2]) && isequal(b, [1 2 3; 1 2 3]));

% ── sub2ind / ind2sub ──────────────────────────────────────────────────
assert(sub2ind([2 3], 1) == 1);
assert(sub2ind([2 3], 2, 3) == 6);
check_err('sub2ind([2 3], 1+1i, 1)', 'MATLAB:sub2ind:RealSubscript', ...
    'Subscript arguments must be real.');
check_err('ind2sub([2 3], 1+1i)', 'MATLAB:ind2sub:RealSubscript', ...
    'Subscript argument must be real.');
[r, c] = ind2sub([2 3 4], 7);
assert(r == 1 && c == 4);

% ── reshape / permute ──────────────────────────────────────────────────
x = reshape(true, 1, 1);
assert(islogical(x) && x);
check_err('reshape(1:4, 2, 1+1i)', 'MATLAB:checkDimCommon:complexSize', ...
    'Size argument cannot be complex.');
x = permute(logical([1 0]), [2 1]);
assert(islogical(x) && isequal(x, logical([1; 0])));
assert(permute(1+1i, [2 1]) == 1+1i);

disp('SUCCESS');

function check_err(expr, id, msg)
    try
        eval([expr ';']);
        error('expected error from %s', expr);
    catch e
        if ~strcmp(e.identifier, id) || ~strcmp(e.message, msg)
            error('%s: got %s / %s', expr, e.identifier, e.message);
        end
    end
end
