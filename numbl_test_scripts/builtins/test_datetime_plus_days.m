% datetime and duration plus/minus a number: the number counts as days.

d = datetime(2020, 1, 1);
assert(d + 1 == datetime(2020, 1, 2));
assert(1 + d == datetime(2020, 1, 2));
assert(d - 1 == datetime(2019, 12, 31));
assert(plus(d, 2) == datetime(2020, 1, 3));
assert(d + 1.5 == datetime(2020, 1, 2, 12, 0, 0));
assert(d + true == datetime(2020, 1, 2));
assert(d - 0.25 == datetime(2019, 12, 31, 18, 0, 0));
assert(isa(d + 1, 'datetime'));
try
    e = 1 - d; %#ok<NASGU>
    error('no error');
catch err
    assert(strcmp(err.message, 'Cannot subtract a datetime array from a double array.'));
end

h = hours(1);
assert(hours(h + 1) == 25);
assert(hours(1 + h) == 25);
assert(hours(h - 1) == -23);
assert(hours(1 - h) == 23);
assert(isa(h + 1, 'duration'));

disp('SUCCESS');
