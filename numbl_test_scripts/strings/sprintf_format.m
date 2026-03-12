% Test sprintf formatting

s1 = sprintf('%d', 42);
assert(strcmp(s1, '42'));

s2 = sprintf('%.2f', 3.14159);
assert(strcmp(s2, '3.14'));

s3 = sprintf('%s and %s', 'hello', 'world');
assert(strcmp(s3, 'hello and world'));

s4 = sprintf('%05d', 7);
assert(strcmp(s4, '00007'));

% num2str
s5 = num2str(123);
assert(strcmp(s5, '123'));

% str2num
n = str2num('42');
assert(n == 42);

% str2double
d = str2double('3.14');
assert(abs(d - 3.14) < 1e-5);

disp('SUCCESS')
