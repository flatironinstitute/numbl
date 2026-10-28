classdef listener < handle
    % A listener for an event, created by addlistener (tied to the source
    % objects' lifetime) or by listener / event.listener (alive while a
    % variable holds it). numbl provides the constructor
    % (interpreter/events.ts).
    properties
        Source = {}
        EventName = ''
        Callback = []
        Enabled (1,1) logical = true
        Recursive (1,1) logical = false
    end
end
