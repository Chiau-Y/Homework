[y, fs]=audioread('D:\Master (2019-2021)\Homework\Digital signal processing\DAC\music.wav');
time=(1:length(y))/fs;	% 時間軸的向量
plot(time, y(:,1));	% 畫出時間軸上的波形
%sound(y,fs);
B=[y(:,1)+1];    % 待匯入的資料（矩陣形式）
xlswrite('D:\Master (2019-2021)\Homework\Digital signal processing\DAC_echo\digital_3.xlsx', B);

