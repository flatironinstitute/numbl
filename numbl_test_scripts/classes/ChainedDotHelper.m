classdef ChainedDotHelper
    properties
        data
    end
    methods
        function obj = ChainedDotHelper()
            obj.data = struct('x', 42, 'y', 99);
        end
        function out = subsref(obj, ind)
            switch ind(1).type
                case '.'
                    out = obj.(ind(1).subs);
                    ind(1) = [];
                    if ~isempty(ind)
                        out = subsref(out, ind);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end
    end
end
