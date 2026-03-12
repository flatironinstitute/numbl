classdef MyPrefClass
    properties (Access = protected)
        prefList
    end
    methods
        function obj = MyPrefClass(techVal)
            obj.prefList = struct();
            obj.prefList.tech = techVal;
            obj.prefList.alpha = 42;
        end
        function out = subsref(obj, ind)
            switch ind(1).type
                case '.'
                    if isfield(obj.prefList, ind(1).subs)
                        out = obj.prefList.(ind(1).subs);
                    else
                        out = builtin('subsref', obj, ind(1));
                    end
                    if numel(ind) > 1
                        out = subsref(out, ind(2:end));
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end
    end
end
