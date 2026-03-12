% Tilde (~) as ignored input parameter

% Call with all three arguments - third is ignored
example_func(1, 2, 99);

% Call with only two arguments since third is irrelevant
example_func(10, 20);

disp('SUCCESS')

function example_func(a, b, ~)
  disp(a + b);
end
