function [all_svm] = Allsvm( X, y, label)

m = size(X, 1);
%n = size(X, 2);

% You need to return the following variables correctly 
all_svm = [];

% Add ones to the X data matrix
%X = [ones(m, 1) X];


for i=1:label
	%svmStruct = svmtrain(X,(y==i),'Kernel_Function','polynomial', 'polyorder', 5);
    svmStruct = svmtrain(X,(y==i),'Kernel_Function', 'rbf', 'rbf_sigma', 1);
	all_svm = [all_svm ; svmStruct];
end


end

