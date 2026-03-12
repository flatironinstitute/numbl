classdef CellHolder
    properties
        funs
    end
    methods
        function obj = CellHolder(funs)
            if nargin > 0
                obj.funs = funs;
            end
        end
        function obj = uminus(obj)
            for k = 1:numel(obj.funs)
                obj.funs{k} = -obj.funs{k};
            end
        end
    end
end
