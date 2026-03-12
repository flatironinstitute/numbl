classdef ConfiguredObj_
    properties
        Speed
        Quality
        Mode
    end
    methods
        function obj = ConfiguredObj_(mode)
            obj.Mode = mode;
            if strcmp(mode, 'fast')
                obj.Speed = 100;
                obj.Quality = 50;
            elseif strcmp(mode, 'quality')
                obj.Speed = 30;
                obj.Quality = 100;
            else
                obj.Speed = 65;
                obj.Quality = 75;
            end
        end
        function r = describe(obj)
            r = obj.Speed + obj.Quality;
        end
        function r = to_counter(obj)
            % Cross-class factory: return a SimpleCounter_ instance
            r = SimpleCounter_(obj.Speed, obj.Quality);
        end
        function r = combine(obj, other)
            % Method that takes another ConfiguredObj_
            total_speed = obj.Speed + other.Speed;
            if total_speed > 150
                r = ConfiguredObj_('fast');
            else
                r = ConfiguredObj_('balanced');
            end
        end
    end
end
