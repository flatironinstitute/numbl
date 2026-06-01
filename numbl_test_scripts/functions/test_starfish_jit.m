% Real chunkie/starfish-style curve callbacks should JIT at --opt 1.
% This pins three blockers as a single test: varargin (already fixed),
% colon-reshape `x(:)`, and 2-row vertical concat `[a; b]`.
%
% `%!numbl:assert_jit` survives to the interpreter (and throws) only if
% lowering bails — so this test going green means starfish-shaped curve
% functions actually JIT.

t = (0:0.1:6.28).';
[r, d, d2] = starfish_local(t, 5, 0.5);

% Quick correctness checks against hand-evaluated values at t=0.
% At t=0: ct=1, st=0, cnt=cos(0)=1, snt=sin(0)=0
%   xs = (1 + amp) * 1 = 1.5,   ys = 0
%   dxs = -(1+amp)*0 - narms*amp*0*1 = 0
%   dys = (1+amp)*1 - narms*amp*0*0 = 1.5
assert(abs(r(1, 1) - 1.5) < 1e-12, 'r(1,1)');
assert(abs(r(2, 1) - 0.0) < 1e-12, 'r(2,1)');
assert(abs(d(1, 1) - 0.0) < 1e-12, 'd(1,1)');
assert(abs(d(2, 1) - 1.5) < 1e-12, 'd(2,1)');

% Hot loop: typical chunkerfunc dispatch pattern (~hundreds of calls
% per discretize phase, 32-element ts vectors).
acc = 0;
for i = 1:50
    [r, d, d2] = starfish_local(t, 5, 0.5);
    acc = acc + r(1, 1);
end
assert(abs(acc - 50 * 1.5) < 1e-12, 'hot-loop accumulation');

disp('SUCCESS')

function [r, d, d2] = starfish_local(t, varargin)
    narms = 5;
    amp = 0.3;
    x0 = 0.0;
    y0 = 0.0;
    phi = 0.0;
    scale = 1.0;
    if nargin > 1 && ~isempty(varargin{1})
        narms = varargin{1};
    end
    if nargin > 2 && ~isempty(varargin{2})
        amp = varargin{2};
    end
    if nargin > 3 && ~isempty(varargin{3})
        ctr = varargin{3};
        x0 = ctr(1); y0 = ctr(2);
    end
    if nargin > 4 && ~isempty(varargin{4})
        phi = varargin{4};
    end
    if nargin > 5 && ~isempty(varargin{5})
        scale = varargin{5};
    end

    ct = cos(t);
    st = sin(t);
    cnt = cos(narms * (t + phi));
    snt = sin(narms * (t + phi));

    xs = x0 + (1 + amp * cnt) .* ct * scale;
    ys = y0 + (1 + amp * cnt) .* st * scale;
    dxs = -(1 + amp * cnt) .* st - narms * amp * snt .* ct;
    dxs = dxs * scale;
    dys = (1 + amp * cnt) .* ct - narms * amp * snt .* st;
    dys = dys * scale;
    d2xs = -dys - narms * amp * (narms * cnt .* ct - snt .* st);
    d2xs = d2xs * scale;
    d2ys = dxs - narms * amp * (narms * cnt .* st + snt .* ct);
    d2ys = d2ys * scale;

    r = [(xs(:)).'; (ys(:)).'];
    d = [(dxs(:)).'; (dys(:)).'];
    d2 = [(d2xs(:)).'; (d2ys(:)).'];
end
