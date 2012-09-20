clear all;

%% Artificial data input (onsets)

% generate on the beat onsets at tempo of 44100
processed = (1:200).*44100;

% add some gaussian noise
std = 44100*0.05;
for i = 1:size(processed, 2)
    processed(i) = processed(i) + randn()*std;
end

%% Initialize

% parameters
lambda = 1;
q = 1;
Rk = 44100;

% constants
H = [1 0];
I = eye(2);

% used by Kalman filter
Pk = zeros([2,2]);
xk = zeros([2,1]);

% initialize this (this may not be an optimal initial state)
oldPk = 44100.*ones([2,2])./2;

% first element: assume that first onset is on the beat
% (this assumption's not necessarily correct in general--think pick-ups)
% second element: guess "randomly" that tempo is 35000
% (obviously we know that it's really 44100, but assume we don't know that)
oldxk = [processed(1) 35000]';

%% Kalman filter

beats = [];
tempo = [];
for k = 2:100
    k;
    yk = processed(k); %onset position
        
    y = 1; % assume onsets are on the beat
    A = [1 y; 0 1];
        
    % Kalman Predict
    Qk = q*[y^3/3 y^2/2; y^2/2 y]; % innovation noise covariance
    Pk(1:end, 1:end) = A*oldPk(1:end, 1:end)*A' + Qk;
    Wk = H*Pk(1:end, 1:end)*H'+Rk; % residual (innovation) covariance
    xk(1:end, 1:end) = A*oldxk(1:end, 1:end); % predicted onset/tempo
        
    % p(yk|y1:k-1,c1:k):
    pyk = normpdf(yk, H*xk, Wk)+eps;
    
    % [onset #, position, position guess, tempo guess, (un-)certainty]
    [k yk/44100 xk(1)/44100 xk(2)/44100 pyk*100000]

    % Kalman Update
    residualError = yk - H*xk(1:end, 1:end);
    Kk = Pk(1:end, 1:end)*H'*(Wk)^(-1); % optimal Kalman gain
    xk(1:end, 1:end) = xk(1:end, 1:end) + Kk*residualError;
    Pk(1:end, 1:end) = (I-Kk*H)*Pk(1:end, 1:end);
    
    oldPk = Pk;
    oldxk = xk;
end
