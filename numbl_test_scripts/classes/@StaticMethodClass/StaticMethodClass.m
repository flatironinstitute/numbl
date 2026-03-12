classdef StaticMethodClass
    properties
        value
    end
    methods
        function obj = StaticMethodClass(v)
            if nargin > 0
                obj.value = v;
            else
                obj.value = 0;
            end
        end
        function r = useStatic(obj)
            r = StaticMethodClass.addOne(obj.value);
        end
    end
    methods (Static = true)
        result = addOne(x);
    end
end
