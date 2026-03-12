classdef MyVals
    properties
        values
        isReal
    end
    methods
        function obj = MyVals(v)
            obj.values = v;
            obj.isReal = true(1, size(v, 2));
        end
    end
end
