classdef AttrValue_
    % Fixture for class_access_jit.m: a value class with a private
    % property (simple enough for the JIT's class support).
    properties
        y = 2
    end
    properties (Access = private)
        x = 1
    end
    methods
        function v = getX(obj)
            v = obj.x;
        end
    end
end
