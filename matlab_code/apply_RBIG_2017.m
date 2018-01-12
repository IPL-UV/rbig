%
% This function apply to the data 'dat' ther transformaction 'Trans' obtained with
% the RBIG.m function. 
% datT is the transformed data.
%
% USE:
%
% [datT] = apply_RBIG(dat,Trans);
%

function [datT] = apply_RBIG_2017(dat,Trans)

DIM = size(dat,1);
Ns = size(dat,2);
precision = Trans(1).precision;

% It is computationally faster to compute the transformation in groups of 500000

Nd = 500000;
mmod = mod(Ns,Nd);
fflor = floor(Ns/Nd);

datT = zeros(size(dat));


for nn=1:Nd:fflor*Nd
    dat0 = dat(:,nn:nn+Nd-1);
    for n = 1:length(Trans)
        % n
        for dim = 1:DIM
            [dat0(dim,:)]= marginal_gaussianizationB(dat0(dim,:),Trans(n).TT(dim).T,precision);
        end

        V = Trans(n).V;
        dat0 = V'*dat0;
    end
    datT(:,nn:nn+Nd-1) = dat0;
end

if mmod>0
        dat0 = dat(:,fflor*Nd+1:end);
    for n = 1:length(Trans)
        % n
        for dim = 1:DIM
            [dat0(dim,:)]= marginal_gaussianizationB(dat0(dim,:),Trans(n).TT(dim).T,precision);
        end

        V = Trans(n).V;
        dat0 = V'*dat0;
    end
    datT(:,fflor*Nd+1:end) = dat0;
end


function [x_gauss] = marginal_gaussianizationB(x,T,precision) 

    if nargin == 2,
        precision = round(sqrt(length(x))+1);
    end

    [x_unif] = marginal_uniformizationB(x,T,precision);

    x_gauss = norminv(x_unif);

function [x_lin] = marginal_uniformizationB(x,T,precision)

    if nargin < 3,
        precision = 1000;
    end

    x_lin = interp1(T.R,T.C,x);
