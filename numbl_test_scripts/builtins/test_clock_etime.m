% Test clock and etime builtins

% Test 1: clock returns 1x6 vector
t = clock;
assert(numel(t) == 6);
[r, c] = size(t);
assert(r == 1);
assert(c == 6);

% Test 2: year is reasonable
assert(t(1) >= 2024);
assert(t(1) <= 2100);

% Test 3: month 1-12
assert(t(2) >= 1 && t(2) <= 12);

% Test 4: day 1-31
assert(t(3) >= 1 && t(3) <= 31);

% Test 5: hour 0-23
assert(t(4) >= 0 && t(4) <= 23);

% Test 6: minute 0-59
assert(t(5) >= 0 && t(5) <= 59);

% Test 7: seconds 0-60
assert(t(6) >= 0 && t(6) < 60);

% Test 8: etime returns elapsed time
t0 = clock;
t1 = clock;
dt = etime(t1, t0);
assert(dt >= 0);
assert(dt < 1); % should be near-instant

disp('SUCCESS');
