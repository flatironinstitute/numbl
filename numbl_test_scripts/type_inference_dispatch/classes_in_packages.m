% Test classes in packages: hard scenarios - cross-class interaction,
% methods from package classes, chained calls, methods taking instances
% of sibling package classes, cross-class construction from script level

% --- Basic construction and methods ---
c = shapes.Circle(5);
assert(strcmp(__inferred_type_str(c), 'ClassInstance<shapes.Circle>'));
assert(c.Radius == 5);
assert(abs(c.area() - pi * 25) < 1e-10);

r = shapes.Rectangle(3, 4);
assert(strcmp(__inferred_type_str(r), 'ClassInstance<shapes.Rectangle>'));
assert(r.area() == 12);
assert(r.perimeter() == 14);

% --- Method returning instance of same package class ---
c2 = c.scale(2);
assert(strcmp(__inferred_type_str(c2), 'ClassInstance<shapes.Circle>'));
assert(c2.Radius == 10);
assert(abs(c2.area() - pi * 100) < 1e-10);

% --- Chain: scale -> method on result ---
assert(abs(c.scale(3).area() - pi * 225) < 1e-10);

% --- Method taking instance of another package class ---
r2 = shapes.Rectangle(12, 12);
assert(c.fits_in_rect(r2), 'circle r=5 fits in 12x12');
r3 = shapes.Rectangle(8, 8);
assert(~c.fits_in_rect(r3), 'circle r=5 does not fit in 8x8');

% --- Cross-class construction from script level ---
% Build a bounding rectangle from a circle
d = c.diameter();
br = shapes.Rectangle(d, d);
assert(br.area() == 100);
assert(br.is_square());

% --- Build inscribed circle from rectangle ---
r4 = shapes.Rectangle(6, 10);
ic = shapes.Circle(r4.min_side() / 2);
assert(ic.Radius == 3);
assert(abs(ic.area() - pi * 9) < 1e-10);

% --- Chain: method result as constructor arg for sibling class ---
assert(abs(shapes.Circle(r4.min_side() / 2).circumference() - 6 * pi) < 1e-10);

% --- Round trip via script-level cross construction ---
c3 = shapes.Circle(4);
d2 = c3.diameter();
sq = shapes.Rectangle(d2, d2);
round_trip = shapes.Circle(sq.min_side() / 2);
assert(round_trip.Radius == 4);

% --- Expression mixing both package classes ---
total = c.area() + r.area() + br.area();
expected = pi * 25 + 12 + 100;
assert(abs(total - expected) < 1e-10);

% --- Scale then use result in another package class ---
big_rect = r.scale(3);
assert(strcmp(__inferred_type_str(big_rect), 'ClassInstance<shapes.Rectangle>'));
assert(big_rect.Width == 9);
assert(big_rect.Height == 12);
ic2 = shapes.Circle(big_rect.min_side() / 2);
assert(strcmp(__inferred_type_str(ic2), 'ClassInstance<shapes.Circle>'));
assert(abs(ic2.Radius - 4.5) < 1e-10);

% --- Loop creating package class instances ---
total_area = 0;
for i = 1:4
    ci = shapes.Circle(i);
    total_area = total_area + ci.area();
end
assert(abs(total_area - 30 * pi) < 1e-10);

% --- Two package classes interacting in a loop ---
total_fit = 0;
for i = 1:5
    ci = shapes.Circle(i);
    ri = shapes.Rectangle(i * 3, i * 3);
    if ci.fits_in_rect(ri)
        total_fit = total_fit + 1;
    end
end
% circle diameter = 2i, rect = 3i x 3i: always fits since 2i <= 3i
assert(total_fit == 5);

disp('SUCCESS')
