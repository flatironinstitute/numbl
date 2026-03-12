classdef StaticSubsref
    properties
        value
    end
    methods (Static = true)
        r = getDefault();
    end
    methods
        function obj = StaticSubsref(v)
            if nargin > 0
                obj.value = v;
            else
                obj.value = [];
            end
        end
        function varargout = subsref(obj, s)
            if isempty(obj)
                varargout = {[]};
                return
            end
            switch s(1).type
                case '.'
                    varargout = {obj.value};
                case '()'
                    varargout = {obj.value};
                otherwise
                    error('Unexpected');
            end
        end
    end
end
