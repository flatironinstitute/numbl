classdef Circle
    properties
        Radius
    end
    methods
        function obj = Circle(r)
            obj.Radius = r;
        end
        function a = area(obj)
            a = pi * obj.Radius^2;
        end
        function c = circumference(obj)
            c = 2 * pi * obj.Radius;
        end
        function r = scale(obj, factor)
            r = shapes.Circle(obj.Radius * factor);
        end
        function r = fits_in_rect(obj, rect)
            % Method taking instance of another package class
            r = (2 * obj.Radius <= rect.Width) && (2 * obj.Radius <= rect.Height);
        end
        function r = diameter(obj)
            r = 2 * obj.Radius;
        end
    end
end
