clear
close all
clc

%% DATA

rng(123);               % set seed for random number generator
n_samples = 1000;       % number of samples
aux_x = randn(1, n_samples);    % auxilary random data, X

x(1, :) = cos(aux_x);   % dim 1 - cosine function
x(2, :) = sinc(aux_x);  % dim 2 - sinc function

x = x + 0.2 * randn(size(x));   % add noise to the data
x = [0.5 0.5; -0.5 0.5] * x;

figure(1);
plot(x(1, :), x(2, :), '.');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% RBIG: Learning the transformation that gaussianizes the data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% default parameters
precision = 1000;
porc = 10;
transformation = 'PCA';
n_layers = 1000;

% get size of the data
[n_dims, n_samples] = size(x);

%% get parameters for the tolerance of m and n


num_repititions = 1000;
ee = zeros(size(num_repititions));

for irep = 1:num_repititions
    % hx 
    [p_x, R_x] = hist(randn(1, round(n_samples)), round(sqrt(n_samples)));

    delta_x = R_x(3) - R_x(2); % get the difference between the bins

    %% Calculate the entropy (MLE estimator with miller-maddow correction)
    c = 0.5 * (sum(delta_x>0) - 1) / sum(delta_x);      % Miller-Maddow Correction
    p_x = p_x / sum(p_x);                         % empirical estimate of the dist.
    idx = p_x~=0;                             
    hx = -sum(p_x(idx) .* log2(p_x(idx))) + c;   % plug-in estimator of the entropy with correction

    % hy
    [p_y, R_y] = hist(randn(1, round(n_samples)), round(sqrt(n_samples)));

    delta_y = R_y(3) - R_y(2); % get the difference between the bins

    %% Calculate the entropy (MLE estimator with miller-maddow correction)
    c = 0.5 * (sum(delta_y>0) - 1) / sum(delta_y);      % Miller-Maddow Correction
    p_y = p_y / sum(p_y);                         % empirical estimate of the dist.
    idy = p_y~=0;                             
    hy = -sum(p_y(idy) .* log2(p_y(idy))) + c;   % plug-in estimator of the entropy with correction

    % ee
    ee(irep) = hy - hx;
    
end

tol_m = mean(ee);
tol_d = std(ee);


%% Loop Through Number of layers

for ilayer = 1:n_layers
    
    tic;
    
    %% Marginal Gaussianization
    for idim = 1:n_dims
        
        p_aux = (porc / 100) * abs(max(x(:)) - min(x(:)));
        R_aux = linspace(min(x(:)), max(x(:)), 2 * sqrt(length(x)) + 1);
        
        R = mean([R_aux(1:end-1); R_aux(2:end)]);
        
        break
        
    end
    
    break
    
    
end
