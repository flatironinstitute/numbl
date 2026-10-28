classdef proplistener < event.listener
    % A listener for a property event (PreSet, PostSet, PreGet, PostGet).
    % Object holds the observed objects and Source their meta.property
    % objects. numbl provides the constructor (interpreter/events.ts).
    properties
        Object = {}
    end
end
