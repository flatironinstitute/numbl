function n = clear_counter_()
% Helper for test_clear_forms.m: counts its calls in a persistent.
persistent c
if isempty(c), c = 0; end
c = c + 1;
n = c;
end
