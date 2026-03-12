function r = normalized(obj)
    % Separate-file method that calls classdef method (magnitude)
    % and another separate-file method implicitly via scale
    m = obj.magnitude();
    r = obj.scale(1 / m);
end
