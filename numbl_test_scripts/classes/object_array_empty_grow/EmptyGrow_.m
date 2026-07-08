classdef EmptyGrow_
    properties
        v = 0
    end
    methods
        function obj = EmptyGrow_(x)
            if nargin > 0
                obj.v = x;
            end
        end
    end
end
