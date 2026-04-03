# Language Features

numbl implements a substantial subset of the MATLAB language.

## Operators

| Category   | Operators                                   |
| ---------- | ------------------------------------------- |
| Arithmetic | `+` `-` `*` `/` `\` `^` `.*` `./` `.\` `.^` |
| Comparison | `==` `~=` `<` `<=` `>` `>=`                 |
| Logical    | `&&` `\|\|` `&` `\|` `~` `xor`              |
| Transpose  | `'` `.'`                                    |

## Data Types

| Type            | Status                                   |
| --------------- | ---------------------------------------- |
| double          | Default numeric type                     |
| logical         | Supported                                |
| char            | Supported (`'text'`)                     |
| string          | Supported (`"text"`)                     |
| complex         | Full support throughout                  |
| cell            | Supported                                |
| struct          | Supported                                |
| sparse          | Supported (real and complex, CSC format) |
| function_handle | Supported                                |
| class instances | Supported (value and handle classes)     |
| dictionary      | Supported (MATLAB R2022b+ feature)       |

Integer types (`int8`, `uint16`, etc.) and single-precision are not supported.

## Control Flow

All standard MATLAB control flow is supported:

```matlab
% if / elseif / else
if x > 0
    disp('positive');
elseif x < 0
    disp('negative');
else
    disp('zero');
end

% for loops
for i = 1:10
    fprintf('%d ', i);
end

% while loops
while n > 1
    n = n / 2;
end

% switch / case
switch color
    case 'red',   code = 1;
    case 'blue',  code = 2;
    otherwise,     code = 0;
end

% try / catch
try
    result = riskyOperation();
catch e
    fprintf('Error: %s\n', e.message);
end
```

`break`, `continue`, and `return` work as expected.

## Functions

```matlab
% Regular function
function result = add(a, b)
    result = a + b;
end

% Multiple return values
[q, r] = quorem(17, 5);

% Anonymous functions
f = @(x) x.^2 + 1;

% Function handles
g = @sin;

% Variable arguments
function out = flexible(varargin)
    out = nargin;
end
```

Nested functions and subfunctions are supported.

## Classes

```matlab
classdef Point
    properties
        x
        y
    end
    methods
        function obj = Point(x, y)
            obj.x = x;
            obj.y = y;
        end
        function d = dist(obj)
            d = sqrt(obj.x^2 + obj.y^2);
        end
    end
    methods (Static)
        function p = origin()
            p = Point(0, 0);
        end
    end
end
```

Inheritance, abstract classes, enumerations, and both value and handle classes are supported.

## Matrix and Array Syntax

```matlab
A = [1 2 3; 4 5 6];       % Matrix literal
v = 1:0.5:10;             % Colon range
A(end, :)                  % end indexing
A(A > 3)                   % Logical indexing
c = {1, 'hello', [1 2]};  % Cell array
s.name = 'test';           % Struct
s.(fieldName)              % Dynamic field access
```

## Comments

```matlab
% Line comment
x = 1; % Inline comment

%{
  Block comment
  spanning multiple lines
%}
```

## Global and Persistent Variables

```matlab
function count()
    persistent n
    if isempty(n)
        n = 0;
    end
    n = n + 1;
    disp(n);
end
```

## Not Supported

Notable MATLAB features not yet implemented:

- Single-precision and integer numeric types
- Parallel computing (`parfor`, `spmd`)
- GPU arrays
- MEX interface
- Simulink and toolboxes
- Java / .NET integration
- App Designer / GUI
- Metaclasses and advanced class introspection
