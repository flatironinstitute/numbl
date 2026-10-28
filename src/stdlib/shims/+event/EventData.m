classdef EventData < handle
    % Data passed to event listener callbacks: the object that triggered
    % the event and the event's name. notify sets both; subclass it to pass
    % more data (notify(obj, 'Name', MyData(...))).
    properties (SetAccess = private)
        Source = []
        EventName = ''
    end
end
