classdef Base
    properties
        v = 0
    end
    methods
        function obj = Base(x)
            if nargin > 0
                obj.v = x;
            end
        end
    end
end
