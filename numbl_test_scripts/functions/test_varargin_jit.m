% Functions with `varargin` should be JIT-eligible at --opt 1 when called
% with concrete typed args. The chunkie/MATLAB-Central idiom of "optional
% scalar params after a primary tensor" is extremely common, and currently
% blocks the call JIT entirely (classifyCall requires
% argTypes.length === fn.params.length).
%
% `%!numbl:assert_jit` inside each callee asserts the surrounding function
% body got JIT-compiled — if lowering bails, the directive survives to the
% interpreter and throws.

% 1) Trailing optional scalars (the starfish-style pattern).
t = (0:0.1:6.28).';
[r, d] = curve_with_opts(t, 5, 0.5);
assert(abs(r(1) - 1.5) < 1e-12, '1: curve_with_opts r(1)');
assert(abs(d(1) - 0.0) < 1e-12, '1: curve_with_opts d(1)');

% Hot loop: the call JIT should kick in well before this finishes.
s = 0;
for i = 1:50
    [r, d] = curve_with_opts(t, 5, 0.5);
    s = s + r(1) + d(1);
end
assert(abs(s - 50 * 1.5) < 1e-12, '1: hot-loop accumulation');

% 2) Same callee with all defaults (zero trailing args).
[r2, d2] = curve_with_opts(t);
assert(abs(r2(1) - 1.3) < 1e-12, '2: defaults r(1)');

% 3) varargin where only some optional args are filled in.
[r3, d3] = curve_with_opts(t, 7);
assert(abs(r3(1) - 1.3) < 1e-12, '3: partial r(1) (narms used, amp default)');

% 4) varargin{i} with a non-literal loop-variable index. V1 lowers only
%    literal-indexed varargin reads, so scaled_sum stays interpreted —
%    it must still produce the correct result.
v = scaled_sum(10, 1, 2, 3);
assert(v == 60, '4: scaled_sum with 3 trailing scalars');

disp('SUCCESS')

function [r, d] = curve_with_opts(t, varargin)
    %!numbl:assert_jit
    narms = 5;
    amp = 0.3;
    if nargin > 1 && ~isempty(varargin{1})
        narms = varargin{1};
    end
    if nargin > 2 && ~isempty(varargin{2})
        amp = varargin{2};
    end
    cnt = cos(narms * t);
    snt = sin(narms * t);
    ct = cos(t);
    st = sin(t);
    r = (1 + amp * cnt) .* ct;
    d = -(1 + amp * cnt) .* st - narms * amp * snt .* ct;
end

function s = scaled_sum(scale, varargin)
    s = 0;
    for i = 1:nargin - 1
        s = s + scale * varargin{i};
    end
end
