classdef ScalarHolder_
    % Simple value class with no subsref/subsasgn — exercises default
    % paren-indexing path.
    properties
        data
    end
    methods
        function obj = ScalarHolder_(v)
            if nargin > 0
                obj.data = v;
            else
                obj.data = [];
            end
        end
    end
end
