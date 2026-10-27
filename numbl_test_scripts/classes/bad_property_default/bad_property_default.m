% A property default that errors makes construction fail (it used to leave
% the property silently unset).
try
    obj = BadDefault_(); %#ok<NASGU>
    error('expected an error');
catch e
    assert(contains(e.message, 'default'));
end
disp('SUCCESS')
