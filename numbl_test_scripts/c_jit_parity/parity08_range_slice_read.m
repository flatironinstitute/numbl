% C-JIT parity gap #08: RangeSliceRead (r = src(a:b)).
%
% The JS-JIT compiles `src(a:b)` on a real tensor via subarrayCopy1r;
% the C-JIT historically bailed feasibility with
%   "unsupported expr: RangeSliceRead"
% because there was no per-iter range-copy helper and no tensor-local
% buffer management for fresh slice results.
%
% Expected disp output (should match across all runs):
%   numbl --opt 1 run <this>                         -> 100\n8280\n-1
%   numbl --opt 2 run <this>                         -> 100\n8280\n-1  (silent JS-JIT fallback)
%   numbl --opt 2 --check-c-jit-parity run <this>    -> 100\n8280\n-1  (the gap is now closed)
%   matlab -batch parity08_range_slice_read          -> 100\n8280\n-1

% 1) Basic range read inside a loop: r = src(a:b), then scalar read
%    r(1). The fresh tensor local `r` must have data/len visible to
%    the subsequent 1-D Index read.
src = [10 20 30 40 50 60 70 80];
s = slice_sum(src, 2, 5);
disp(s)   % 5 iters; r(1) = src(2) = 20 each iter; sum = 100.

% 2) Range read with dynamic endpoints that vary each call. Two
%    disjoint slices summed over 20 iters.
base = [101 102 103 104 105 106];
t = two_slice_sum(base, 3);
disp(t)   % per iter: 101 + 103 + 104 + 106 = 414; times 20 = 8280.

% 3) Out-of-bounds start is a hard bounds error (err-flag 1.0 in C).
threw = false;
try
    oob_read();
catch
    threw = true;
end
assert(threw, 'OOB slice read should throw');
disp(-1)

function s = slice_sum(src, a, b)
    s = 0;
    for k = 1:5
        r = src(a:b);
        s = s + r(1);
    end
end

function t = two_slice_sum(base, dim)
    M = 2 * dim;
    t = 0;
    for k = 1:20
        r0 = base(1:dim);
        r1 = base(dim+1:M);
        t = t + r0(1) + r0(dim) + r1(1) + r1(dim);
    end
end

function oob_read()
    oob = zeros(1, 5);
    r = oob(0:3); %#ok<NASGU>
end
