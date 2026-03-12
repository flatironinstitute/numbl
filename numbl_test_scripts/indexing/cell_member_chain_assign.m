% Test chained cell+member assignment: data{1}.field = val
% where data starts uninitialized (as an output variable)

function data = make_data()
  data{1}.name = 'trig';
  data{1}.value = 42;
end

result = make_data();
assert(strcmp(result{1}.name, 'trig'));
assert(result{1}.value == 42);

% Test growing cell with chained assignment
function data = make_data2()
  data{1}.x = 10;
  data{2}.x = 20;
end

d = make_data2();
assert(d{1}.x == 10);
assert(d{2}.x == 20);

% Test multi-level member chain: data{1}.a.b = val
function data = make_data3()
  data{1}.a.b = 99;
end

d3 = make_data3();
assert(d3{1}.a.b == 99);

disp('SUCCESS');
