% lookup(d, keys, 'FallbackValue', fb) returns fb for missing keys
% (MATLAB R2023b+ dictionary API).

d = configureDictionary("string", "double");
d("alpha") = 1;
d("beta") = 2;

% Plain lookup still works
assert(lookup(d, "alpha") == 1);

% Fallback used for a missing key
assert(lookup(d, "gamma", 'FallbackValue', 0) == 0);
assert(lookup(d, "gamma", 'FallbackValue', -5) == -5);

% Fallback ignored when the key exists
assert(lookup(d, "beta", 'FallbackValue', 99) == 2);

% Method-call syntax (as used by code storing dictionaries in properties)
assert(d.lookup("alpha", 'FallbackValue', 0) == 1);
assert(d.lookup("missing", 'FallbackValue', 7) == 7);

% Missing key without fallback still errors
ok = false;
try
    lookup(d, "nope");
catch
    ok = true;
end
assert(ok);

disp('SUCCESS');
