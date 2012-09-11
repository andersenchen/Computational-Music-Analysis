%%Load music here!
[y, Fs] = wavread('ciara.wav');
oneChannel = y(:, 1);
% oneChannel = zeros(1,size(y,1));
% newSize = int64(size(y,1));
% for i = 1:newSize
%     oneChannel(i) = y(i,1);
% end
%% FFT, processing
ySize = size(y,1);
windowSize = 1024;
fSize = 1024;
nWindows = int64(ySize/windowSize*2);
hammingWindow = hamming(windowSize);
hannWindow = hann(windowSize);
timeWindow = zeros(1,windowSize);
for i = 1:windowSize
    timeWindow(i) = 1.5*sin(2*pi/(windowSize)*(i-windowSize/2));
end
loltest = [1:512];
transformWindow = transferFunction(loltest*44);
c1 = 5;
c2 = 348;

sumsArray = zeros(1, nWindows-1);
middleSumsArray = zeros(1, nWindows-1);
difsArray = zeros(1, nWindows-2);
prevArray = zeros(1, c2);
middleNormalSums = zeros(1, nWindows-1);
weightedFinalSums = zeros(1, nWindows-1);
difsArray2 = zeros(1, nWindows-2);
for i = 1:nWindows-2
    start = int64((i-1)*windowSize*0.5)+1;
    myWindow = oneChannel(start:(start+windowSize)-1);
    myWindow1 = myWindow'.*hammingWindow;
    myWindow2 = myWindow'.*timeWindow';
    myWindow3 = myWindow'.*hannWindow;
    timefft = fft(myWindow2);
    hannfft = fft(myWindow3);
    newfft = timefft./hannfft;
    newfftReal = real(newfft);
    middleSumsArray(i) = sum(newfftReal(c1+1:c2));
    four = fft(myWindow1);
    four = abs(four);
    sumsArray(i) = sum(four(1:c1))+sum(four(c2+1:512));
    middleNormalSums(i) = sum(four(c1+1:c2))+middleSumsArray(i);
    
end

for i = 2:size(sumsArray, 2)-1
    difsArray(i-1) = (sumsArray(i+1)-sumsArray(i)) + (sumsArray(i)-sumsArray(i-1));
    difsArray2(i-1) = (middleNormalSums(i+1)-middleNormalSums(i)) + (middleNormalSums(i)-middleNormalSums(i-1));
end
%% Generate graphs and stuff
raw = (difsArray');
raw1 = difsArray2';
n = size(raw);
n1 = size(raw1);
onsets2 = zeros(n, 2);
onsets3 = zeros(n, 2);
for i = 1:n;
    onsets2(i,1) = i;
    onsets2(i,2) = raw(i);
    onsets3(i,1) = i;
    onsets3(i,2) = raw1(i);
end
processed = [];
average = 10;
for i = average+1:n-average
    A = onsets2(i-average:i+average, 2);
    M = mean(A);
    amin = min(A);
    amax = max(A);
    B = onsets3(i-average:i+average, 2);
    M1 = mean(B);
    amin = min(B);
    amax = max(B); 
    C = middleNormalSums(i-average:i+average);%mean(onsets2(i-1:i+1,2)) > M+1*std(A) || 
    if  mean(onsets2(i-1:i+1,2)) > M+1*std(A) || (mean(onsets3(i-1:i+1,2)) > M1+1.25*std(B));
        processed = [processed i*windowSize/2];
    end
end
toRemove = [];
size(processed)
for i = 2:size(processed,2)
    if processed(i) - processed(i-1) < 5000;
        toRemove = [toRemove i];
    end
end
processed(toRemove) = [];
size(processed)
%% MIR stuff, shortcut meybes?
mirverbose(0);
lol = mironsets('ciara.wav', 'Detect', 'Peaks');
data = mirgetdata(lol);
processed = (floor(data*Fs))';
%% 
tempChannel = oneChannel;
lengthS = 1000/44100;
t1 = 0:1:44100*lengthS;
y1 = 3/4*sin((2*pi/(44100/440))*t1);
y2 = y1*4;
st1 = size(t1,2);
sth = int64(st1/2);
for i = 1:size(processed,2)
    for j = 1:st1
        ind = processed(i)+j-250;
        tempChannel(ind) = tempChannel(ind)*0.25;
        tempChannel(ind) = tempChannel(ind)+y1(j);
    end
end

%% Write .wav file
wavwrite(tempChannel, Fs, 'onsetOutput.wav');

