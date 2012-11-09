clear all;

%% Train the classifier
[classifier, tclass] = train_joint();

%% Load music here!
[y, Fs] = wavread('chord.wav');
full = zeros(1,size(y,1));
newSize = int32(size(y,1));
for i = 1:newSize
    full(i) = y(i,1);
end

% might not want to load whole thing if it's long
%oneChannel = full(1,1:Fs*20);
oneChannel = full(1,:);

% FFT, processing
ySize = size(oneChannel,2);
windowSize = 2^12;
nWindows = int32(ySize/windowSize*2);
hannWindow = hann(windowSize);
spectrum = zeros(nWindows,windowSize);
truespectrum = zeros(size(spectrum));

for i = 1:nWindows-2
    start = int32((i-1)*windowSize/2)+1;
    myWindow = oneChannel(start:(start+windowSize)-1);
    myWindow1 = myWindow'.*hannWindow;
    truespectrum(i,:) = fft(myWindow1);
    spectrum(i,:) = abs(fft(myWindow1));
end

%% NMF BITCH
tic
spectra = 0;
coeff = 0;
argmin = 0;
min = Inf;

for i = 8
    
    i
    [spectra, coeff] = nmfsc(spectrum' ,i, [], 0.6, 'wat', 0);
    entropy = sum(sum(spectrum'-(spectra*coeff)).^2)/i*sqrt(i);
    if entropy < min
        argmin = i;
        min = entropy;
    end
end
argmin
min

[spectra, coeff] = nnmf(spectrum' ,8);%, 'w0', classifier');


%, 'w0', classifier');
toc
%%
firstNote = spectra(:, 8);
in = ifft(firstNote);
in = repmat(in, 10, 1);
player = audioplayer(in, 44100);
play(player);

%% Pseudoinverse solution

M = pinv(classifier)';

spectra = zeros(nWindows,16);
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));

    % pseudoinverse solution
    spectra(i,:) = M*u';
end

%% NMF solution (multiplicative rule)

spectra = zeros(nWindows,16);
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));

    % NMF
    spectra(i,:) = .5*ones(16,1);
    for k = 1:20 % until convergence
        spectra(i,:) = (spectra(i,:)'.*(classifier*(spectrum(i,:)'./(classifier'*spectra(i,:)')))) ...
                       ./(classifier*ones(size(spectrum(i,:)')));
    end
end

%%
plot(1 - gampdf(1:1:500,2,100)/max(gampdf(1:1:500,2,100)));

%%

x = (1:1:800) / 250;
mu = 1;
sigma = .5;

plot(1./((x)*pi*sigma.*(1 + ((log(x) - mu)/sigma).^2)))

%%
for i = 2:10
    abs(spectra(i,:) - spectra(i-1,:))
end
    
%% gradient descent solution (additive)

spectra = ones(nWindows,16);
for i = 1:nWindows
    % normalize to unit vector
    u = spectrum(i,:) / sum(spectrum(i,:));
 
    spectra(i,:) = zeros(16,1);
    eps = 50;

    for k = 1:100 % until convergence
        if (i == 1)
            spectra(i,:) = spectra(i,:) + ...
                eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)')';
        else
            kgam = 2;
            theta = 1;
            
            % gamma distribution does opposite of what we want
            %dpxx1 = theta^(-1) - (kgam-1)*(abs(spectra(i,:) - spectra(i-1,:)) + .01).^(-1);
            
            % log cauchy distribution
            x = (abs(spectra(i,:) - spectra(i-1,:)) + 5) / 250;
            dpxx1 = sigma*(mu^2 - 2*mu + sigma^2 - 2*(mu - 1)*log(x) + (log(x)).^2) ...
                ./ (pi*(x.^2).*(mu^2 + sigma^2 - 2*mu*log(x) + (log(x)).^2).^2);
            
            spectra(i,:) = spectra(i,:) + ...
                eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)' - dpxx1')';
        end
        
%        spectra(i,:) = spectra(i,:) + ...
%            eps*(classifier*spectrum(i,:)' - classifier*classifier'*spectra(i,:)' + spectra(i,:)')';

        % constraint enforcement; project back into domain
        for m = 1:size(spectra,2)
            if spectra(i,m) < 0
                spectra(i,m) = 0;
            end
        end

        % sparsity
        %if k > 50
        %    for m = 1:size(spectra,2)
        %        if spectra(i,m) < 300
        %            spectra(i,m) = 0;
        %        end
        %    end
        %end

        if eps > 2
            eps = eps - 1;
        else
            eps = .90*eps;
        end
    end
end

%% Graph output

output = fliplr(spectra); % to print right
surf([output zeros(nWindows,1)]); % need extra padding for surf (wtf)

%% Generate spectra time lapse
for i = 1:nWindows
    plot(spectra(i,:))
    pause(windowSize/Fs);
end

%% Generate an interpretive output (won't work; boundary issue insurmountable)
newChannel = zeros(size(oneChannel));
lastWindow = ifft(tclass'*spectra(1,:)');
max(lastWindow)
min(lastWindow)
for i = 2:nWindows-1
    start = int32((i-1)*windowSize/2)+1;
    myWindow = ifft(tclass'*spectra(i,:)');
    myWindow1 = lastWindow(windowSize/2+1:end) + myWindow(1:windowSize/2);
    newChannel(start:(start+windowSize/2)-1) = myWindow1;
    lastWindow = myWindow;
end

% mastering
newChannel = newChannel / max(newChannel);

% Write .wav file
wavwrite(newChannel, Fs, 'gen.wav');
