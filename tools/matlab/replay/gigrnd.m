function X = gigrnd(p, a, b, sampleSize)
% gigrnd - replay version for the tests: the mode of the generalized inverse Gaussian
% distribution with density proportional to x^(p-1) exp(-(a x + b/x)/2), in place of a draw;
% no uniform is used. Python draws from SciPy's geninvgauss, whose algorithm differs from that
% of third_party/gigrnd.m, so the replay versions in both languages return the mode.
if nargin < 4
    sampleSize = 1;
end
X = repmat(((p - 1) + sqrt((p - 1)^2 + a*b)) / a, sampleSize, 1);
end
