classdef OpMultiplier_
    properties
        Base
    end
    methods
        function obj = OpMultiplier_(b)
            obj.Base = b;
        end
        function r = apply_op(obj, x)
            r = obj.Base * x;
        end
        function r = describe_op(obj)
            r = 2;
        end
    end
end
