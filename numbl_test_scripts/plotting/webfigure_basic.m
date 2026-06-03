% webfigure: build a directory figure from a containers.Map and emit it.
% Runs headless here (no rendering); this exercises the Map traversal, the
% webfigure.m wrapper, and the drawwebfigure native emit primitive.

d = containers.Map();
d('index.html') = ['<!doctype html><html><body><canvas id="c" width="300" height="200"></canvas>' ...
    '<script src="app.js"></script></body></html>'];
d('app.js') = ['fetch("data.json").then(r=>r.json()).then(D=>{' ...
    'const g=document.getElementById("c").getContext("2d");' ...
    'D.x.forEach((xi,i)=>g.fillRect(xi*3,200-D.y[i]*3,3,3));});'];
d('data.json') = '{"x":[0,10,20,30],"y":[10,40,20,50]}';

webfigure(d);

disp('SUCCESS')
