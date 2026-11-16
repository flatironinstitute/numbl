% Colormap functions return m-by-3 RGB tables (256 rows by default) whose
% values match MATLAB's. Reference values for m = 5 were produced by MATLAB
% R2026b. parula is excluded: numbl approximates it.

names = {'jet', 'hsv', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'gray', 'bone', 'copper', 'pink'};

% Sizes, including the default and edge cases
for k = 1:numel(names)
  f = str2func(names{k});
  assert(isequal(size(f()), [256 3]), [names{k} ': default size']);
  assert(isequal(size(f(8)), [8 3]), [names{k} ': size for m = 8']);
  assert(isequal(size(f(1)), [1 3]), [names{k} ': size for m = 1']);
  assert(isequal(size(f(0)), [0 3]), [names{k} ': size for m = 0']);
  c = f(64);
  assert(all(c(:) >= 0 & c(:) <= 1), [names{k} ': values in [0, 1]']);
end

% Exact values for m = 5
ref = [0.0 0.5 1.0; 0.0 1.0 1.0; 0.5 1.0 0.5; 1.0 1.0 0.0; 1.0 0.5 0.0];
assert(max(abs(jet(5) - ref), [], 'all') < 1e-12, 'jet(5) values');
ref = [1.0 0.0 0.0; 0.7999999999999998 1.0 0.0; 0.0 1.0 0.40000000000000036; 0.0 0.40000000000000036 1.0; 0.8000000000000007 0.0 1.0];
assert(max(abs(hsv(5) - ref), [], 'all') < 1e-12, 'hsv(5) values');
ref = [1.0 0.0 0.0; 1.0 1.0 0.0; 1.0 1.0 0.3333333333333333; 1.0 1.0 0.6666666666666666; 1.0 1.0 1.0];
assert(max(abs(hot(5) - ref), [], 'all') < 1e-12, 'hot(5) values');
ref = [0.0 1.0 1.0; 0.25 0.75 1.0; 0.5 0.5 1.0; 0.75 0.25 1.0; 1.0 0.0 1.0];
assert(max(abs(cool(5) - ref), [], 'all') < 1e-12, 'cool(5) values');
ref = [1.0 0.0 1.0; 1.0 0.25 0.75; 1.0 0.5 0.5; 1.0 0.75 0.25; 1.0 1.0 0.0];
assert(max(abs(spring(5) - ref), [], 'all') < 1e-12, 'spring(5) values');
ref = [0.0 0.5 0.4; 0.25 0.625 0.4; 0.5 0.75 0.4; 0.75 0.875 0.4; 1.0 1.0 0.4];
assert(max(abs(summer(5) - ref), [], 'all') < 1e-12, 'summer(5) values');
ref = [1.0 0.0 0.0; 1.0 0.25 0.0; 1.0 0.5 0.0; 1.0 0.75 0.0; 1.0 1.0 0.0];
assert(max(abs(autumn(5) - ref), [], 'all') < 1e-12, 'autumn(5) values');
ref = [0.0 0.0 1.0; 0.0 0.25 0.875; 0.0 0.5 0.75; 0.0 0.75 0.625; 0.0 1.0 0.5];
assert(max(abs(winter(5) - ref), [], 'all') < 1e-12, 'winter(5) values');
ref = [0.0 0.0 0.0; 0.25 0.25 0.25; 0.5 0.5 0.5; 0.75 0.75 0.75; 1.0 1.0 1.0];
assert(max(abs(gray(5) - ref), [], 'all') < 1e-12, 'gray(5) values');
ref = [0.0 0.0 0.125; 0.21875 0.34375 0.34375; 0.4791666666666667 0.5625 0.5625; 0.7395833333333334 0.78125 0.78125; 1.0 1.0 1.0];
assert(max(abs(bone(5) - ref), [], 'all') < 1e-12, 'bone(5) values');
ref = [0.0 0.0 0.0; 0.3125 0.1953 0.124375; 0.625 0.3906 0.24875; 0.9375 0.5859 0.373125; 1.0 0.7812 0.4975];
assert(max(abs(copper(5) - ref), [], 'all') < 1e-12, 'copper(5) values');
ref = [0.5773502691896257 0.0 0.0; 0.7071067811865476 0.7071067811865476 0.408248290463863; 0.816496580927726 0.816496580927726 0.6666666666666666; 0.9128709291752769 0.9128709291752769 0.8498365855987975; 1.0 1.0 1.0];
assert(max(abs(pink(5) - ref), [], 'all') < 1e-12, 'pink(5) values');

% parula(m) still has the right shape
assert(isequal(size(parula(7)), [7 3]));

disp('SUCCESS')
