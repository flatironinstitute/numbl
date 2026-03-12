classdef BasicObj_
    properties
        val
    end
    methods
        function obj = BasicObj_(v)
            obj.val = v;
        end
        function r = plus(a, b)
            % Returns 'basic' to identify which plus was called
            r = 'basic';
        end
        function r = minus(a, b)
            r = 'basic';
        end
        function r = mtimes(a, b)
            r = 'basic';
        end
    end
end
