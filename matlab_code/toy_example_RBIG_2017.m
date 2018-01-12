

clear all
close all

%% 
% 
% addpath(genpath('/media/disk/users/valero/FUMADAS/RBIG_2017/'))

%% DATA
rng(123)
Ns = 1000;
aux_X = randn(1,Ns);

X(1,:) = cos(aux_X);
X(2,:) = sinc(aux_X);

X = X + 0.2*randn(size(X));
X = [0.5 0.5;-0.5 0.5]*X;

figure
plot(X(1,:),X(2,:),'.')

%% RBIG. Learning the transformation that gaussianizes the data 

[datT, Trans, PARAMS] = RBIG_2017(X);

figure
plot(datT(1,:),datT(2,:),'.')
axis equal

%% APPLY learned transformation to new data

[datT2] = apply_RBIG_2017(X,Trans);

sum(sum((datT-datT2).^2))

%% SYNTHESZING new data from "Gaussian" data

dat_rnd = randn(2,Ns);
[dat2] = inv_RBIG_2017(dat_rnd,Trans);

figure
plot(X(1,:),X(2,:),'.')
hold on
plot(dat2(1,:),dat2(2,:),'r.')

legend('original','synthetic')

