classdef Summer
    properties
        data
    end
    methods
        function obj = Summer(d)
            obj.data = d;
        end
        % Method with the same name as the builtin max().
        function out = max(obj, other)
            out = Summer(obj.data + other);
        end
        % Method with the same name as the builtin sum().
        function out = sum(obj)
            out = obj.data * 10;
        end
    end
end
