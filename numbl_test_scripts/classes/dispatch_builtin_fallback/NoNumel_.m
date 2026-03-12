classdef NoNumel_
    % A class that does NOT define numel — calls to numel(obj)
    % should fall back to the built-in numel function.
    properties
        data
    end
    methods
        function obj = NoNumel_(val)
            obj.data = val;
        end
        function out = plus(a, b)
            out = NoNumel_(a.data + b.data);
        end
    end
end
