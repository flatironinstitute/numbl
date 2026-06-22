% svd / norm on non-finite input must match MATLAB: return NaN (or Inf for
% norm of an Inf-but-no-NaN matrix) instead of erroring in LAPACK dgesdd.

% --- svd of a matrix containing NaN -> all-NaN singular values ---
s = svd([NaN 1; 2 3]);
assert(all(isnan(s)));
assert(length(s) == 2);

% --- svd of a matrix containing Inf -> all-NaN singular values (as in MATLAB) ---
s = svd([Inf 0; 0 1]);
assert(all(isnan(s)));

% --- [U,S,V] = svd(NaN matrix): NaN factors, with MATLAB-shaped sizes ---
% (S is the diagonal matrix, so its off-diagonal entries are 0, not NaN.)
[U, S, V] = svd(nan(2, 3));
assert(isequal(size(U), [2 2]));
assert(isequal(size(S), [2 3]));
assert(isequal(size(V), [3 3]));
assert(all(isnan(U(:))));
assert(all(isnan(diag(S))));
assert(all(isnan(V(:))));

% --- economy svd of a NaN matrix ---
[U, S, V] = svd(nan(3, 2), 'econ');
assert(isequal(size(U), [3 2]));
assert(isequal(size(S), [2 2]));
assert(isequal(size(V), [2 2]));
assert(all(isnan(U(:))) && all(isnan(diag(S))) && all(isnan(V(:))));

% --- complex svd with non-finite entries -> NaN (requires LAPACK addon) ---
haveComplexSvd = true;
try
    svd(complex([1 0; 0 1], [1 0; 0 1]));
catch e
    if ~isempty(strfind(e.message, 'requires LAPACK'))
        haveComplexSvd = false;
    else
        rethrow(e);
    end
end
if haveComplexSvd
    s = svd(complex(nan(2, 2), zeros(2, 2)));
    assert(all(isnan(s)));
    s = svd(complex(ones(3, 2), inf(3, 2)));
    assert(all(isnan(s)));

    % Matrix 2-norm of a complex matrix = largest singular value: it must use
    % the complex SVD, not just the real part (which would give ~5.465 here).
    A = complex([1 2; 3 4], [1 1; 1 1]);
    assert(abs(norm(A, 2) - max(svd(A))) < 1e-10);
    assert(abs(norm(A, 2) - 5.8208206) < 1e-6);
end

% --- matrix 2-norm: NaN dominates, else Inf, else finite ---
% (These short-circuit before svd, so they work without the complex addon.)
assert(isnan(norm(nan(3, 10))));        % any NaN  -> NaN
assert(isnan(norm([NaN 0; 0 1])));      % any NaN  -> NaN
assert(norm([Inf 0; 0 1]) == Inf);      % Inf only -> Inf
assert(norm([-Inf 0; 0 1]) == Inf);     % Inf only -> Inf
assert(isnan(norm([Inf NaN; 0 1])));    % Inf + NaN -> NaN
assert(norm(complex([Inf 0; 0 1], zeros(2, 2))) == Inf);

% --- finite matrices still compute correctly ---
A = [4, 3; 2, 1];
s = svd(A);
assert(abs(s(1) - 5.4649857042) < 1e-6);
assert(abs(norm(A, 2) - s(1)) < 1e-10);

disp('SUCCESS');
