classdef Dog_
    properties
        name
    end
    methods
        function obj = Dog_(n)
            if nargin > 0
                obj.name = n;
            else
                obj.name = '';
            end
        end
    end
end
