classdef PrecedenceClass
    properties
        val
    end
    methods
        function obj = PrecedenceClass(v)
            obj.val = v;
        end
        function r = compete(obj)
            r = obj.val;
        end
    end
end
