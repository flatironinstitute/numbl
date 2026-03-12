classdef Rectangle
    properties
        Width
        Height
    end
    methods
        function obj = Rectangle(w, h)
            obj.Width = w;
            obj.Height = h;
        end
        function a = area(obj)
            a = obj.Width * obj.Height;
        end
        function p = perimeter(obj)
            p = 2 * (obj.Width + obj.Height);
        end
        function r = is_square(obj)
            r = obj.Width == obj.Height;
        end
        function r = scale(obj, factor)
            r = shapes.Rectangle(obj.Width * factor, obj.Height * factor);
        end
        function r = min_side(obj)
            r = min(obj.Width, obj.Height);
        end
    end
end
