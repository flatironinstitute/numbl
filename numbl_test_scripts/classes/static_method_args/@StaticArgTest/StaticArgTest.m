classdef StaticArgTest
    methods (Static)
        function result = add(a, b)
            result = a + b;
        end
        function T = process(col, row)
            disp(col);
            T = length(col);
        end
    end
end
