[y, fs]=audioread('D:\Master_2019_2021\Homework\Digital signal processing\DAC\music.wav');
y2 = downsample(y,2);
y3 = downsample(y,4);
time=(1:length(y2))/fs;	% �ɶ��b���V�q
plot(time, y2(:,1));	% �e�X�ɶ��b�W���i��
%sound(y3,fs/4)

e = zeros(1,length(y2));
d = 3 * (fs/2);

for i = (d+1) : 85419 
    e(i) = 0.6*e(i-d) + y2(i,2);
end
%sound(0.2*y2,fs/2)
sound(0.7*e,fs/2)

% B=[y2(:,1)*128+128];    % �ݶפJ����ơ]�x�}�Φ��^
% xlswrite('D:\Master_2019_2021\Homework\Digital signal processing\DAC_echoo\digital_downsample_2.xlsx', B);
% C=[y3(:,1)*128+128];    % �ݶפJ����ơ]�x�}�Φ��^
% xlswrite('D:\Master_2019_2021\Homework\Digital signal processing\DAC_echoo\digital_downsample_4.xlsx', C);
