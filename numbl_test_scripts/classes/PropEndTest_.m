classdef PropEndTest_
    properties
        data
    end
    methods
        function obj = PropEndTest_(d)
            if nargin > 0
                obj.data = d;
            else
                obj.data = [];
            end
        end
    end
end
