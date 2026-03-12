classdef CompoundAssignObj
    properties
        data
    end
    methods
        function obj = CompoundAssignObj()
            obj.data = [];
        end
        function obj = subsasgn(obj, s, val)
            if numel(s) == 1 && strcmp(s(1).type, '()')
                % This means a decomposed store-back triggered subsasgn
                % with just () type - this should not happen for compound
                % V(i).field = rhs assignments
                error('CompoundAssignObj:subsasgn', ...
                    'subsasgn called with single () - store-back should use builtin');
            end
            obj = builtin('subsasgn', obj, s, val);
        end
    end
end
