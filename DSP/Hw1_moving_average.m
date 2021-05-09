clc;
clear;
num = 40;

x = [-num:num];
u = [zeros(1,num),ones(1,10),zeros(1,num-10+1)];   %unit step
subplot(4,1,1),stem(x,u),title('x[n] = u[n] - u[n-10]');

h = 0.9 .^ x .*[zeros(1,num),ones(1,num+1)];   %h[n]
subplot(4,1,2),stem(x,h),title('h[n] = 0.9^n * u[n]');
h2 = fliplr(h);   %鏡射於y軸
subplot(4,1,3),stem(x,h2),title('h[-n]');

y2 = zeros(1,36);
for i = -5:30
    y2(i+6) = sum(circshift(h2,i) .* u);   %shift後相乘再加總
end
t = [-5:30];
subplot(4,1,4),stem(t,y2),title('y[n] = sigma(x[n] * h[n-k])');

