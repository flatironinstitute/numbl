classdef FuncCallDispatchTarget_
    properties
        value
    end
    methods
        function obj = FuncCallDispatchTarget_(v)
            obj.value = v;
        end
        function obj = add_to_value(obj, x)
            obj.value = obj.value + x;
        end
    end
end
