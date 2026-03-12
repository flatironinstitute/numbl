classdef BraceTest
    properties
        data
    end
    methods
        function obj = BraceTest(d)
            obj.data = d;
        end
        function result = subsref(obj, S)
            if strcmp(S(1).type, '{}')
                % Return the sum of the brace indices as a simple test
                subs = S(1).subs;
                result = 0;
                for k = 1:length(subs)
                    result = result + subs{k};
                end
            elseif strcmp(S(1).type, '.')
                result = builtin('subsref', obj, S);
            else
                result = builtin('subsref', obj, S);
            end
        end
    end
end
