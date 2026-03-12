classdef (Abstract) BaseCalc
    properties
        value
    end
    methods (Abstract, Static)
        result = compute(x, y)
    end
    methods
        function obj = BaseCalc(v)
            obj.value = v;
        end
        % Declared stub for external method populate.m
        [obj, r] = populate(obj, x, y)
    end
end
