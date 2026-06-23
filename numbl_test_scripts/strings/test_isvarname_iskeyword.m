% isvarname / iskeyword.
assert(isvarname('abc'), 'plain name');
assert(~isvarname('1abc'), 'leading digit');
assert(isvarname('a_b3'), 'letters digits underscore');
assert(~isvarname('for'), 'keyword is not a valid varname');
assert(~isvarname('a b'), 'space');
assert(~isvarname(''), 'empty');
assert(~isvarname(5), 'non-text');

assert(iskeyword('while'), 'while is a keyword');
assert(~iskeyword('foo'), 'foo is not a keyword');
k = iskeyword();
assert(iscell(k) && any(strcmp(k, 'for')) && any(strcmp(k, 'classdef')), ...
    'iskeyword() returns the keyword list');
disp('SUCCESS');
