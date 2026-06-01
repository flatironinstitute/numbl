classdef TypedPropClass_
    % Properties using MATLAB's property validation syntax
    % (Name (dims) Type {validators} = default), R2019b+.
    properties
        Value double = 7
        Scale (1, 1) double = 2.5
        Tag {mustBeNumeric} = 9
        Plain = 'hello'
    end
end
