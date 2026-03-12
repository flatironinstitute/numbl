classdef OpClass
    properties
        val
    end
    methods
        function obj = OpClass(v)
            obj.val = v;
        end
        function r = minus(a, b)
            if isa(a, 'OpClass')
                av = a.val;
            else
                av = a;
            end
            if isa(b, 'OpClass')
                bv = b.val;
            else
                bv = b;
            end
            r = OpClass(av - bv);
        end
        function r = plus(a, b)
            if isa(a, 'OpClass')
                av = a.val;
            else
                av = a;
            end
            if isa(b, 'OpClass')
                bv = b.val;
            else
                bv = b;
            end
            r = OpClass(av + bv);
        end
        function r = norm(obj, varargin)
            r = abs(obj.val);
        end
        function r = sum(obj)
            r = obj.val;
        end
        function r = abs(obj)
            r = OpClass(abs(obj.val));
        end
    end
end
