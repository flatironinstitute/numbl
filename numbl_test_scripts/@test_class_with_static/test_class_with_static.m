classdef test_class_with_static
    methods (Static)
        function y = example_static(x)
            y = x + 1;
        end
    end
end
