classdef FevalTarget_
% Simple class with a custom feval method for testing feval(obj, ...) dispatch.
    properties
        data
    end
    methods
        function obj = FevalTarget_(d)
            if nargin > 0
                obj.data = d;
            else
                obj.data = [];
            end
        end
        function val = feval(obj, idx)
            % Custom feval: return data at the given index
            val = obj.data(idx);
        end
    end
end
