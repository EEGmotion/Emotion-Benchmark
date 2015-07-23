%% features Takahashi deap mats

meanAllDeap = cell(32,1);
stdAllDeap = cell(32,1);
meanabsdiffAllDeap = cell(32,1);
meanabsdiffnorAllDeap = cell(32,1);
meanabs2diffAllDeap = cell(32,1);
meanabs2diffnorAllDeap = cell(32,1);
EEGfeatures = [];
GSRfeatures = [];

labelAllDeap = cell(32,1);
for i = 1:32
    if i < 10
        load(['../mats_deap/s0' num2str(i) '.mat']);
    else
        load(['../mats_deap/s' num2str(i) '.mat']);
    end
    
    % subject labels
    labelAllDeap{i} = labels;
    % mean
    meanSuj = mean(data,3);
    meanAllDeap{i} = meanSuj;
    % std
    stdSuj = std(data,0,3);
    stdAllDeap{i} = stdSuj; 
    % mean of differences
    meanabsdiff = mean(abs(diff(data,1,3)),3);
    meanabsdiffAllDeap{i} = meanabsdiff;
    % mean of normalized differences
    meanabsdiffnor = meanabsdiff./stdSuj;
    meanabsdiffnorAllDeap{i} = meanabsdiffnor;
    % mean of 2nd differences
    meanabs2diff = mean(abs(diff(data,2,3)),3);
    meanabs2diffAllDeap{i} = meanabs2diff;
    % mean of normalized 2nd differences
    meanabs2diffnor = meanabs2diff./stdSuj;
    meanabs2diffnorAllDeap{i} = meanabs2diffnor;
    
    EEGfeatures = [EEGfeatures; meanSuj(:,1:32) stdSuj(:,1:32) ...
        meanabsdiff(:,1:32) meanabsdiffnor(:,1:32) meanabs2diff(:,1:32) ...
        meanabs2diffnor(:,1:32)];
    GSRfeatures = [GSRfeatures; meanSuj(:,37) stdSuj(:,37) ...
        meanabsdiff(:,37) meanabsdiffnor(:,37) meanabs2diff(:,37) ...
        meanabs2diffnor(:,37)]; 
    clear data labels meanSuj i stdSuj meanabs2diffnor ... 
        meanabs2diff meanabsdiffnor meanabsdiff;
    
end

% features
features = [EEGfeatures GSRfeatures];

%% normalization

meanfea = mean(features);
stdfea = std(features);
features_norm = zeros(size(features));

for i = 1:size(features,2)
    features_norm(:,i) = (features(:,i) - meanfea(i))/stdfea(i);
end

%% SVM Takahashi

%% SVM one-against-all one-leave-out 3 classes

classes_3_train = labels_3(index_final_train_3); 
N = length(classes_3_train);
pred = zeros(N,1);

for i = index_final_train_3
    itrain = index_final_train_3;
    itest = i;
    itrain(find(itest==itrain)) = [];
    smvStructs = Allsvm(features(itrain,:),labels_3(itrain),3);
    p = predictSVM(smvStructs,features(itest,:));
    pred(i,1) = mean(double(p == labels_3(itest))) * 100;
    display([num2str(i) ': ' num2str(pred(i,1)) ' - ' num2str(mean(pred))]);
    [C,order] = confusionmat(double(labels_3(itest)),p);
end

res = mean(pred);

% final testing accuracy

smvStructs = Allsvm(features_norm(index_final_train_3,:),labels_3(index_final_train_3),3);
p = predictSVM(smvStructs,features_norm(index_final_test_3,:));
accuracy_final_3 = mean(double(p == labels_3(index_final_test_3))) * 100;
[C,order] = confusionmat(double(labels_3(index_final_test_3)),p)

%% SVM one-against-all one-leave-out 5 classes

classes_5_train = labels_5(index_final_train_5); 
N = length(classes_5_train);
pred = zeros(N,1);

for i = index_final_train_5
    itrain = index_final_train_5;
    itest = i;
    itrain(find(itest==itrain)) = [];
    smvStructs = Allsvm(features(itrain,:),labels_5(itrain),5);
    p = predictSVM(smvStructs,features(itest,:));
    pred(i,1) = mean(double(p == labels_5(itest))) * 100;
    display([num2str(i) ': ' num2str(pred(i,1)) ' - ' num2str(mean(pred))]);
    [C,order] = confusionmat(double(labels_5(itest)),p);
end

res = mean(pred);

% final testing accuracy

smvStructs = Allsvm(features_norm(index_final_train_5,:),labels_5(index_final_train_5,1),5);
p = predictSVM(smvStructs,features_norm(index_final_test_5,:));
accuracy_final_5 = mean(double(p == labels_5(index_final_test_5))) * 100;
[C,order] = confusionmat(double(labels_5(index_final_test_5)),p)