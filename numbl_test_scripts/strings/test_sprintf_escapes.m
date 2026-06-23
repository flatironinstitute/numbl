% sprintf interprets the standard C escape sequences.
assert(double(sprintf('\n')) == 10, 'newline');
assert(double(sprintf('\t')) == 9, 'tab');
assert(double(sprintf('\r')) == 13, 'carriage return');
assert(double(sprintf('\a')) == 7, 'bell');
assert(double(sprintf('\b')) == 8, 'backspace');
assert(double(sprintf('\f')) == 12, 'form feed');
assert(double(sprintf('\v')) == 11, 'vertical tab');
assert(strcmp(sprintf('\\'), '\'), 'backslash');
assert(length(sprintf('a\rb')) == 3, 'escape is a single char');
disp('SUCCESS');
