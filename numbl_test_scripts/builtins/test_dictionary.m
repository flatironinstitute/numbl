% Test dictionary type

% Create empty dictionary
d = dictionary;
assert(~isConfigured(d));

% Create dictionary with key-value pairs
d = dictionary("Monocycle", 1, "Bicycle", 2, "Tricycle", 3);
assert(numEntries(d) == 3);
assert(isConfigured(d));

% Lookup by key
assert(d("Tricycle") == 3);
assert(d("Monocycle") == 1);

% Modify existing entry
d("Bicycle") = 2.5;
assert(d("Bicycle") == 2.5);

% Add new entry
d("Car") = 4;
assert(numEntries(d) == 4);
assert(d("Car") == 4);

% Remove entry
d("Car") = [];
assert(numEntries(d) == 3);

% isKey
assert(isKey(d, "Bicycle"));
assert(~isKey(d, "Car"));

% keys and values
k = keys(d);
v = values(d);
assert(length(v) == 3);

% configureDictionary
d2 = configureDictionary("string", "double");
assert(isConfigured(d2));
assert(numEntries(d2) == 0);
d2("x") = 10;
assert(d2("x") == 10);

% dictionary with key-value pairs
d3 = dictionary("a", 1, "b", 2, "c", 3);
assert(d3("b") == 2);
assert(numEntries(d3) == 3);

% insert, lookup, remove functions
d5 = dictionary("a", 1);
d5 = insert(d5, "b", 2);
assert(lookup(d5, "b") == 2);
d5 = remove(d5, "a");
assert(numEntries(d5) == 1);
assert(~isKey(d5, "a"));

% entries with two outputs
[k2, v2] = entries(d3);

% types
t = types(d3);

% class
assert(strcmp(class(d3), 'dictionary'));

% Method-style calls
assert(d3.numEntries() == 3);
assert(d3.isConfigured());
assert(d3.isKey("a"));

% Duplicate keys: last value wins
d6 = dictionary("x", 1, "x", 2);
assert(d6("x") == 2);

% Curly-brace indexing with cell values
d7 = dictionary;
d7{"num"} = 42;
d7{"str"} = "hello";
assert(d7{"num"} == 42);

% Numeric keys
d8 = dictionary(1, "one", 2, "two", 3, "three");
assert(strcmp(d8(2), "two"));
assert(numEntries(d8) == 3);

disp('SUCCESS');
