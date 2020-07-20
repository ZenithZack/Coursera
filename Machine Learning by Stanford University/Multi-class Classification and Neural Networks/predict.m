function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%This way helps me with my imagination of a Neural Network
% X = [ones(m, 1) X];
% t1 = sigmoid((Theta1 * X')');
% t1 = [ones(m, 1) t1];

% t2 = sigmoid(Theta2 * t1');

% [~, p] = max(t2, [], 1);

% X = 5000*401, Theta1 = 25*401; L2 = 5000*25; Theta2 = 10*26 

X = [ones(m,1) X];
L2 = sigmoid(X * Theta1');
L2 = [ones(m,1) L2];

L3 = sigmoid(L2 * Theta2');
[~, p] = max(L3, [], 2);

% =========================================================================


end
