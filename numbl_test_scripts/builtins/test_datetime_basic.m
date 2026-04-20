% Test datetime builtin basic forms

% datetime('now') returns a datetime
tn = datetime('now');
assert(strcmp(class(tn), 'datetime'))

% datetime('today') has zero time components
tt = datetime('today');
assert(tt.Hour == 0)
assert(tt.Minute == 0)
assert(tt.Second == 0)

% datetime(Y, M, D)
t1 = datetime(2024, 3, 15);
assert(t1.Year == 2024)
assert(t1.Month == 3)
assert(t1.Day == 15)
assert(t1.Hour == 0)
assert(t1.Minute == 0)
assert(t1.Second == 0)

% datetime(Y, M, D, H, MI, S)
t2 = datetime(2024, 3, 15, 10, 30, 45);
assert(t2.Year == 2024)
assert(t2.Month == 3)
assert(t2.Day == 15)
assert(t2.Hour == 10)
assert(t2.Minute == 30)
assert(t2.Second == 45)

% datetime(X, 'ConvertFrom', 'datenum')
% MATLAB serial date 739252 == 2024-01-01
t3 = datetime(739252, 'ConvertFrom', 'datenum');
assert(t3.Year == 2024)
assert(t3.Month == 1)
assert(t3.Day == 1)

% datetime(X, 'ConvertFrom', 'posixtime')
% POSIX 1704067200 == 2024-01-01 00:00:00 UTC
t4 = datetime(1704067200, 'ConvertFrom', 'posixtime');
assert(strcmp(class(t4), 'datetime'))
assert(t4.Year == 2024)
assert(t4.Month == 1)
assert(t4.Day == 1)

% datetime - datetime produces a duration; seconds() extracts secs.
a = datetime(2024, 1, 1, 0, 0, 0);
b = datetime(2024, 1, 1, 0, 0, 30);
d = b - a;
assert(strcmp(class(d), 'duration'))
assert(seconds(d) == 30)

% datetime + duration / datetime - duration
c = a + seconds(90);
assert(c.Minute == 1)
assert(c.Second == 30)
e = b - seconds(15);
assert(e.Second == 15)

% duration +/- duration
dd = seconds(10) + seconds(5);
assert(seconds(dd) == 15)

% comparisons
assert(b > a)
assert(a < b)
assert(a ~= b)

disp('SUCCESS')
