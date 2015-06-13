function p = predictSVM( all_svm, X )

m = size(X, 1);
num_labels = size(all_svm, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
newClasses = zeros(size(X, 1), size(all_svm, 1));

for i = 1:num_labels
    newClasses(:,i) = svmclassify(all_svm(i),X);
end

for i = 1:m
    if(~isempty(find(newClasses(i,:)==1)))
        p(i) = find(newClasses(i,:)==1,1);
    else
        p(i) = 3;
    end
end

end

