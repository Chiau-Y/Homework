clc;
clear;

% Because the initial condition is 0, for n<0, and y[0] - y[-1] - 0.9y[-2] = x[0], 
% whether the input is Impulse Response or Step Response, 
% y[0] = 1, y[-1] = 0, and y[-2] = 0. 

% --------------- x[n] = Impulse Response ---------------
y = [0 0 1];   % y[-1] = 0, y[-2] = 0, and y[0] = 1
y1 = zeros(1,20);
y2 = [y zeros(1,98)]; 

for i = 1:97
    y2(i+3) = y2(i+2) + 0.9 * y2(i+1);   % x[n] = 0, when n is not equal to 1
end

y3 = [y1 y2];
t = [-20:100];
subplot(2,1,1),stem(t,y3),title('x[n] = Impulse Response');

% --------------- x[n] = Step Response ---------------
yy = [0 0 1];   % y[-1] = 0, y[-2] = 0, and y[0] = 1
yy1 = zeros(1,20);
yy2 = [yy zeros(1,98)];

for i = 1:97
    yy2(i+3) = yy2(i+2) + 0.9 * yy2(i+1) + 1;   % x[n] = 1, when n > 0 
end

yy3 = [yy1 yy2];
t = [-20:100];
subplot(2,1,2),stem(t,yy3),title('x[n] = Step Response');

