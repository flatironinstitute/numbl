classdef Pair_
    properties
        A
        B
    end
    methods
        function obj = Pair_(a, b)
            obj.A = a;
            obj.B = b;
        end
        function r = sum(obj)
            r = obj.A + obj.B;
        end
        function r = swap(obj)
            r = Pair_(obj.B, obj.A);
        end
        function r = doubled(obj)
            r = Pair_(obj.A * 2, obj.B * 2);
        end
    end
end
