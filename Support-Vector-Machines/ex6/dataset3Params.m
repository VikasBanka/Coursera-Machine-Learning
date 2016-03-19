function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

values_C = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
values_Sigma =[0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
predError = zeros(length(values_C),length(values_Sigma));
for i = 1: length(values_C)
    C = values_C(i);
    for j = 1 : length(values_Sigma)
        sigma = values_Sigma(j);
        model = svmTrain(X,y,C, @(x1,x2) gaussianKernel(x1,x2,sigma));
        predictions = svmPredict(model,Xval);
        predError(i,j) = mean(double(predictions ~= yval));
    end
end

[min_sigma,min_c] = find(predError ==min(min(predError)));
C = values_C(min_c);
sigma = values_Sigma(min_sigma);

% =========================================================================

end
