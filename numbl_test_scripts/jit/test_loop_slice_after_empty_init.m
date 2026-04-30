% Slice assignment to a variable that was previously initialized to [].
%
% Mirrors the chunkie adapgausskerneval pattern (line 91-94):
%   datat1 = [];
%   if ~isempty(datat)
%       datat1 = datat(:, ii);
%   end
%   ... use datat1 ...
%
% Before fix: tryLowerAsSliceBind hard-bailed because datat1 was already
% in `assignedVars` (from `datat1 = []`). The bail propagated to the
% caller, blocking the surrounding `for ii = 1:ntarg` from JIT-compiling.
%
% After fix: the slice-alias bind returns null instead of "bail",
% letting normal lowerIndexExpr handle the slice via __extractSlice2d.

m = 8;
n = 16;
A = zeros(m, n);
for r = 1:m
    for c = 1:n
        A(r, c) = r * 100 + c;
    end
end

% Pattern A: non-empty source, if-branch executes.
% We pass the slice into a helper to mirror the chunkie use of `datat1`
% as an argument to `oneintp(...)`. Returning a scalar avoids the
% downstream-bail noise from sum(unknown-shape).
total = 0;
for i = 1:n
    %!numbl:assert_jit
    x = [];
    if ~isempty(A)
        x = A(:, i);
    end
    total = total + helper_first(x);
end

% Expected: sum of A(1, i) for i = 1..n  →  100 * n + (1+2+...+n)
expected = 100 * n + n * (n + 1) / 2;
assert(total == expected, 'Pattern A: scalar accumulator');

% Pattern B: empty source, if-branch skipped.
B = [];
total2 = 0;
n2 = 5;
for i = 1:n2
    %!numbl:assert_jit
    y = [];
    if ~isempty(B)
        y = B(:, i);
    end
    total2 = total2 + helper_count(y);
end
assert(total2 == 0, 'Pattern B: empty branch only');

disp('SUCCESS');

function v = helper_first(x)
    if isempty(x)
        v = 0;
    else
        v = x(1);
    end
end

function v = helper_count(x)
    v = numel(x);
end
