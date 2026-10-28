classdef EvPayload < event.EventData
    % Fixture for the events tests: event data carrying a payload.
    properties
        Payload
    end
    methods
        function d = EvPayload(p)
            d.Payload = p;
        end
    end
end
