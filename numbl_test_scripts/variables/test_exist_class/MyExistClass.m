classdef MyExistClass
    properties
        value
    end
    methods
        function obj = MyExistClass(v)
            if nargin > 0
                obj.value = v;
            end
        end
    end
end
