classdef StaticFalseCtorTest_
    properties
        value = 0
        initialized = 0
    end
    methods (Access = public, Static = false)
        function obj = StaticFalseCtorTest_(v)
            if nargin > 0
                obj.value = v;
            end
            obj.initialized = 1;
        end
    end
    methods
        function result = plus(a, b)
            result = StaticFalseCtorTest_(a.value + b.value);
        end
    end
end
