% Test computer builtin

% Basic call returns a char vector
s = computer;
assert(ischar(s));
assert(~isempty(s));

% Architecture string
a = computer('arch');
assert(ischar(a));
assert(~isempty(a));

% Two outputs: str and maxsize
[s2, maxsz] = computer;
assert(ischar(s2));
assert(strcmp(s, s2));
assert(maxsz == 2^48 - 1);

% Three outputs: str, maxsize, endian
[s3, maxsz2, endian] = computer;
assert(strcmp(s, s3));
assert(maxsz2 == 2^48 - 1);
assert(strcmp(endian, 'L'));

% On Linux: should be GLNXA64 / glnxa64
if isunix && ~ismac
    assert(strcmp(s, 'GLNXA64'));
    assert(strcmp(a, 'glnxa64'));
end

disp('SUCCESS');
