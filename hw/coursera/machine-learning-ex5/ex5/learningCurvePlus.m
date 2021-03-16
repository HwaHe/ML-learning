function [error_train, error_val] = ...
  learningCurvePlus(X, y, Xval, yval, lambda, freq=50)
%LEARNINGCURVEPLUS Generates the train and cross validation set errors needed to
%to plot a learning curve, the difference from the `learningCurve` function is 
%that this function will calculate the average error value with argument `freq`.
%   [error_train, error_val] = ...
%       LEARNINGCURVEPLUS(X, y, Xval, yval, lambda, freq) returns the train and 
%       cross validation set errors for a learning curve. In particular, it
%       returns two vectors of the same length - error_train and error_val.
%       Then error_train(i) contains the training error for i examples (and
%       similarly for error_val(i)).
%

% Number for training examples
m = size(X, 1);

% the return values
error_train = zeros(m, 1);
error_val = zeros(m, 1);

% logic code here
for i = 1:m,
  for j=1:freq,
    % calculate train error over randomly selected i samples
    train_sel = randperm(m, i);
    theta_train = trainLinearReg(X(train_sel, :), y(train_sel, :), lambda);
    error_train(i)+=linearRegCostFunction(X(train_sel, :), y(train_sel, :), ... 
                                      theta_train, 0);
    
    % calculate cross validation error over randomly selected i samples using 
    % theta learned from above i training samples.
    val_sel = randperm(m, i);
    error_val(i) += linearRegCostFunction(Xval(val_sel, :), yval(val_sel, :), ...
                                      theta_train, 0);
  endfor
  
  % calculate the average error values
  error_train(i)/=freq;
  error_val(i)/=freq;
endfor

end