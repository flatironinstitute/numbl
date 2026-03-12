classdef MyTech
    properties
        value
    end
    methods
        function obj = MyTech(val)
            if nargin < 1
                obj.value = 42;
            else
                obj.value = val;
            end
        end
    end
end
