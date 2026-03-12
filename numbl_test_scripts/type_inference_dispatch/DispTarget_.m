classdef DispTarget_
    properties
        Val
    end
    methods
        function obj = DispTarget_(v)
            obj.Val = v;
        end
        function r = transform(obj, x)
            % Class method: multiply
            r = obj.Val * x;
        end
        function r = compute(obj)
            r = obj.Val + 1000;
        end
    end
end
