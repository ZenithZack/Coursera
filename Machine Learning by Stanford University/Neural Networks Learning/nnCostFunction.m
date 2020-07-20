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
%Theta1 = 25*401; Theta2 = 10*26

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

% FORWARD PROPAGATION
X = [ones(m,1) X]; %5000 * 401
Z2 = Theta1 * X';   %(25 * 401) * (401 * 5000)
A2 = sigmoid(Z2);  %(25 * 5000)

A2 = [ones(m,1) A2']; %(5000 * 26)
Z3 = Theta2 * A2'; % (10 * 5000)
h_theta = sigmoid(Z3); %h_theta is the same as A3 or OUTPUT LAYER - (10 * 5000)

y_new = zeros(num_labels,m); % 10 * 5000
for i = 1 : m
	y_new(y(i), i) = 1;
end

% The first "sum" is for summing up the errors of each image (result = 1 * 5000 - column-wise-sum)
% The second "sum" is for summing up the errors of all images (result = 1 * 1)
J = (1/m) * sum(sum((-y_new) .* log(h_theta) - (1 - y_new) .* log(1 - h_theta))); 

% Bias terms are not supposed to be regularized
t1 = Theta1(:, 2:size(Theta1,2));
t2 = Theta2(:, 2:size(Theta2,2));

% REGULARIZATION
Reg = (lambda/(2 * m)) * (sum(sum(t1.^2)) + sum(sum(t2.^2)));

% Regularized Cost Function
J = J + Reg; 
%
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

%Theta1 = 25*401; Theta2 = 10*26

% BACKPROPOAGATION
for img = 1:m
	
	% STEP 1 - Forward propagating one image
	A1 = X(img,:); % A1 = (1*401)
	A1 = A1'; % A1 = (401*1)
	Z2 = Theta1 * A1; % Z2 = (25*1)
	A2 = sigmoid(Z2);
	
	A2 = [1 ; A2]; % A2 = (26*1)
	Z3 = Theta2 * A2;
	A3 = sigmoid(Z3); %A3 = (10*1) (Also called the output layer)
	
	% STEP 2 
	delta_3 = A3 - y_new(:,img); %delta_3 = (10*1)
	
	%STEP 3
	Z2 = [1; Z2];
	delta_2 = (Theta2' * delta_3) .* sigmoidGradient(Z2); %delta_2 = (16*1)
	delta_2 = delta_2(2:end);
	
	%STEP 4
	Theta2_grad = Theta2_grad + (delta_3 * A2'); % Theta2_grad = (10*26) = (10*1)*(1*26)
	Theta1_grad = Theta1_grad + (delta_2 * A1'); % Theta1_grad = (25*401) = (25*1)*(1*401)

end;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta2_grad = (1/m) * Theta2_grad; % Theta2_grad = (10*26)
Theta1_grad = (1/m) * Theta1_grad; %Theta1_grad = (25*401)

Theta1_grad(:, 2:size(Theta1,2)) = Theta1_grad(:, 2:size(Theta1,2)) + ((lambda/m) * Theta1(:, 2:size(Theta1,2)));
Theta2_grad(:, 2:size(Theta2, 2)) = Theta2_grad(:, 2:size(Theta2, 2)) + ((lambda/m) * Theta2(:, 2:size(Theta2, 2)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
