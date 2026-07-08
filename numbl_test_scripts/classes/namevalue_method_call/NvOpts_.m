classdef NvOpts_
    properties
        base = 1
    end
    methods
        function obj = NvOpts_(b)
            obj.base = b;
        end
        function r = scale(obj, x, opts)
            arguments
                obj
                x
                opts.factor = 1
            end
            r = obj.base * x * opts.factor;
        end
    end
    methods (Static)
        function r = combine(x, opts)
            arguments
                x
                opts.offset = 0
            end
            r = x + opts.offset;
        end
    end
end
