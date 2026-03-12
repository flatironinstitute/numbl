classdef MyIndexedObj
    properties
        data
    end
    methods
        function obj = MyIndexedObj(d)
            obj.data = d;
        end
        function obj = fixup(obj)
            for k = 1:numel(obj)
                obj(k).data = obj(k).data + 1;
            end
        end
        function varargout = subsref(obj, s)
            switch s(1).type
                case '()'
                    % Numeric indexing evaluates something
                    out = obj.data;
                    if numel(s) > 1
                        out = builtin('subsref', out, s(2:end));
                    end
                case '.'
                    out = builtin('subsref', obj, s);
                otherwise
                    error('Unsupported indexing');
            end
            varargout = {out};
        end
        function obj = subsasgn(obj, s, val)
            switch s(1).type
                case '.'
                    obj = builtin('subsasgn', obj, s, val);
                otherwise
                    error('Unsupported assignment');
            end
        end
    end
end
