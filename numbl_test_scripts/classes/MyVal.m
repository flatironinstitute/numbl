classdef MyVal
    properties
        data
    end
    methods
        function obj = MyVal(d)
            obj.data = d;
        end
        function obj = add(obj, val)
            obj.data = obj.data + val;
        end
    end
end
