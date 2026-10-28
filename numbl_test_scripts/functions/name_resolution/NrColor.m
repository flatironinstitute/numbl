classdef NrColor
    enumeration
        Red, Green, Blue
    end
    methods (Static)
        function c = pick(k)
            all = [NrColor.Red, NrColor.Green, NrColor.Blue];
            c = all(k);
        end
    end
end
