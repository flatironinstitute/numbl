classdef MathUtils_
    properties
        Value
    end
    methods
        function obj = MathUtils_(val)
            obj.Value = val;
        end
        function r = get_value(obj)
            r = obj.Value;
        end
        function r = add_to_value(obj, x)
            r = obj.Value + x;
        end
        function r = apply_static(obj)
            % Instance method calling static method on own class
            r = MathUtils_.square(obj.Value);
        end
    end
    methods (Static)
        function r = square(x)
            r = x * x;
        end
        function r = double_it(x)
            r = x * 2;
        end
        function r = quad(x)
            r = MathUtils_.square(x) * 4;
        end
        function obj = create(val)
            obj = MathUtils_(val);
        end
        function r = sum_values(a, b)
            obj1 = MathUtils_.create(a);
            obj2 = MathUtils_.create(b);
            r = obj1.get_value() + obj2.get_value();
        end
    end
end
