classdef ConstColor_
    % Fixture for class_constants.m: an enumeration with a Constant property.
    enumeration
        Red, Green
    end
    properties (Constant)
        Default = ConstColor_.Green
    end
end
