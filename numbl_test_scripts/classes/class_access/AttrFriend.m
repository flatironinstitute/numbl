classdef AttrFriend
    % Fixture for class_access.m: a class granted access by
    % Access = {?AttrFriend}.
    methods (Static)
        function v = peek(c)
            v = c.friendly;
        end
        function poke(c, v)
            c.friendly = v;
        end
        function v = peekPriv(c)
            v = c.priv;
        end
    end
end
