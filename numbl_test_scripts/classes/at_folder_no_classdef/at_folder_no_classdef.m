% Test: @ClassName methods without a classdef should not shadow regular functions.
%
% @UnknownClass/greet.m exists but UnknownClass has no classdef file.
% greet.m also exists as a regular workspace function.
% Calling greet() should resolve to the regular function, not the @-folder method.

result = greet('world');
assert(strcmp(result, 'hello world'), ...
    ['Expected "hello world", got "' result '"']);

disp('SUCCESS')
