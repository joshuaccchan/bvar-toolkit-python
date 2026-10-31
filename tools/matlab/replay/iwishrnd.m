function W = iwishrnd(Tau, df)
% iwishrnd - replay version for the tests: W = inv(C*A*A'*C') with C = chol(inv(Tau),
% 'lower') and A lower triangular, whose diagonal holds the square roots of n chi-square
% draws with df, df - 1, ..., df - n + 1 degrees of freedom and whose n(n-1)/2 entries
% below it are normal draws, taken in that order and filled column by column.
n = size(Tau, 1);
A = diag(sqrt(gamrnd((df - (0:n-1)')/2, 2)));
low = tril(true(n), -1);
A(low) = randn(nnz(low), 1);
CA = chol(inv(Tau), 'lower')*A;
W = inv(CA*CA');
end
