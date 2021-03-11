function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% cost function
% comment shorthands:
%   ils: input_layer_size
%   hls: hidden_layer_size
%   nl:  num_labels
bias=ones(m,  1);  % m x 1
A1=[bias X];  % m x (ils+1)
Z2=A1*(Theta1)';  % m x hls
A2=[bias sigmoid(Z2)];  % m x (hls+1)
Z3=A2*(Theta2)';  % m x nl
A3=sigmoid(Z3);  % m x nl

Y=[];
for i=1:m
    Y(:, i)=(1:num_labels==y(i))';
end

% Y nl x m

% cost term
cost = 1/m*sum((-Y .* log((A3)') - (1-Y).*log(1-(A3)'))(:));

% regularization term
theta=lambda/(2*m)*(sum((Theta1(:, 2:end).**2)(:))+sum((Theta2(:, 2:end).**2)(:)));

% loss J
J=cost + theta;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% Derivatives values of Theta1 & Theta2
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

% Vectorization operations
delta3=A3'-Y;  % nl x m
delta2=Theta2'*delta3.*[zeros(m, 1) sigmoidGradient(Z2)]'; % (hls+1) x m


Delta1=delta2(2:end, :)*A1;  % hls x (ils+1)
Delta2=delta3*A2;  % nl x (hls+1)

Theta1_grad=Delta1 ./ m;
Theta2_grad=Delta2 ./ m;


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda/m*Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda/m*Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
