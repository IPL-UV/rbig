%
% Function that computes the inverse transformation (Trans)
% obtained by RBIG over the data (dat).
%
% USE:
% [dat2] = inv_RBIG(datT,Trans)
%

function [dat2] = inv_RBIG_2017(datT,Trans)


precision = Trans(1).precision;
DIM = size(datT,1);
dat2  =datT;

for n = length(Trans):-1:1

        V = Trans(n).V;
        dat2 = V*dat2;

    for dim = 1:DIM
            [dat2(dim,:)]= inv_marginal_gaussianization(dat2(dim,:),Trans(n).TT(dim).T,precision);
    end
end

function [x2] = inv_marginal_gaussianization(x_gauss,T,precision)

if nargin == 2, precision = round(sqrt(length(x_gauss))+1); end
x_lin = normcdf(x_gauss);
[x2] = inv_marginal_uniformization(x_lin,T,precision);

function [x2] = inv_marginal_uniformization(x_lin,T,precision)

if nargin == 2, precision = 1000; end
x2 = interp1(T.C,T.R,x_lin);
