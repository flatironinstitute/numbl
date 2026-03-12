classdef ScalarMultiIdx_
    properties
        val
    end
    methods
        function obj = ScalarMultiIdx_(v)
            obj.val = v;
        end
        function varargout = subsref(obj, S)
            if strcmp(S(1).type, '()')
                % Delegate to builtin for () indexing on scalar
                [varargout{1:nargout}] = builtin('subsref', obj, S);
                return
            end
            [varargout{1:nargout}] = builtin('subsref', obj, S);
        end
    end
end
