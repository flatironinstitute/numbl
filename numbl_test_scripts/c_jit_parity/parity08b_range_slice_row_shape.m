% RangeSliceRead must preserve the source's row/column orientation.
%
% Regression for a JIT bug where `src(a:b)` on a row-vector source
% emitted a column-vector slice; shape-sensitive downstream ops
% (notably isequal) then misclassified matching contents as unequal,
% so a search loop like find_signature returned -1 under --opt 1
% while the interpreter returned the correct index.

buf = double(1:100);
buf(50) = 500; buf(51) = 501; buf(52) = 502; buf(53) = 503;
sig = [500 501 502 503];
assert(isequal(size(buf), [1 100]), 'buf should be a row vector');

% Row-vector slice compared with a row-vector sig via isequal, in a
% loop with `break`. The JIT compiles this loop; a column-shaped
% slice would make isequal always return false.
m = numel(sig);
n = numel(buf);
found = -1;
for k = n - m + 1:-1:1
    if isequal(buf(k:k+m-1), sig)
        found = k;
        break
    end
end
assert(found == 50, sprintf('row-vector find failed: found=%d', found));

% Column-vector source: slice stays column (baseline).
bufC = buf';
sigC = sig';
assert(isequal(size(bufC), [100 1]), 'bufC should be a column vector');
found = -1;
for k = n - m + 1:-1:1
    if isequal(bufC(k:k+m-1), sigC)
        found = k;
        break
    end
end
assert(found == 50, sprintf('column-vector find failed: found=%d', found));

disp('SUCCESS')
