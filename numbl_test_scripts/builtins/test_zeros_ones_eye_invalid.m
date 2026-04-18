% zeros / ones / eye must reject non-integer and NaN sizes.
% MATLAB silently clamps negatives to 0 (empty result) -- match that.
threw = false;
try; zeros(2.5); catch; threw = true; end
if ~threw, error('zeros(2.5) should have errored'); end

threw = false;
try; ones(2.5); catch; threw = true; end
if ~threw, error('ones(2.5) should have errored'); end

threw = false;
try; eye(2.5); catch; threw = true; end
if ~threw, error('eye(2.5) should have errored'); end

threw = false;
try; zeros(NaN); catch; threw = true; end
if ~threw, error('zeros(NaN) should have errored'); end

% negatives silently produce empty (matches MATLAB)
z = zeros(-2);
assert(isempty(z));
assert(size(z,1) == 0 && size(z,2) == 0);

o = ones(-3);
assert(isempty(o));

% Inf is rejected
threw = false;
try; zeros(Inf); catch; threw = true; end
if ~threw, error('zeros(Inf) should have errored'); end

% valid still work
assert(isequal(zeros(2), [0 0; 0 0]));
assert(isequal(ones(2), [1 1; 1 1]));
assert(isequal(eye(2), [1 0; 0 1]));

disp('SUCCESS');
