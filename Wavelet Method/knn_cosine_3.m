%% 3 classes

pos = label_valencia(:,1)>=6.33333;
neg = label_valencia(:,1)<3.66667;
neu = label_valencia(:,1)<6.33333 & label_valencia(:,1)>=3.66667;

labels = [];
labels(pos) = 1;
labels(neu) = 3;
labels(neg) = 2;
labels = labels';

%% KNN REE, K=5, Cosine.
features=EEG_REE';
N = length(labels);

load('indices_test_3.mat')

itest=indice; 
itrain = setdiff(1:N,itest)';

cp1 = classperf(labels);
knn = knnclassify(features(itest,:),features(itrain,:),labels(itrain),5,'cosine');
classperf(cp1,knn,itest);

accuracy1=cp1.correctRate;
error1=cp1.errorRate;
sensitivity1=cp1.Sensitivity;
specificity1=cp1.Specificity;

%% KNN REE, K=10, Cosine.

cp2 = classperf(labels);
knn = knnclassify(features(itest,:),features(itrain,:),labels(itrain),10,'cosine');
classperf(cp2,knn,itest);

accuracy2=cp2.correctRate;
error2=cp2.errorRate;
sensitivity2=cp2.Sensitivity;
specificity2=cp2.Specificity;

%% KNN LREE, K=5, Ecuclidean.
features=EEG_LREE';

cp3 = classperf(labels);
knn = knnclassify(features(itest,:),features(itrain,:),labels(itrain),5,'cosine');
classperf(cp3,knn,itest);

accuracy3=cp3.correctRate;
error3=cp3.errorRate;
sensitivity3=cp3.Sensitivity;
specificity3=cp3.Specificity;

%% KNN LREE, K=10, Cosine.

cp4 = classperf(labels);
knn = knnclassify(features(itest,:),features(itrain,:),labels(itrain),10,'cosine');
classperf(cp4,knn,itest);

accuracy4=cp4.correctRate;
error4=cp4.errorRate;
sensitivity4=cp4.Sensitivity;
specificity4=cp4.Specificity;

%% KNN ALREE, K=5, Cosine.
features=EEG_ALREE';


cp5 = classperf(labels);
knn = knnclassify(features(itest,:),features(itrain,:),labels(itrain),5,'cosine');
classperf(cp5,knn,itest);

accuracy5=cp5.correctRate;
error5=cp5.errorRate;
sensitivity5=cp5.Sensitivity;
specificity5=cp5.Specificity;

%% KNN ALREE, K=10, Cosine.

cp6 = classperf(labels);
knn = knnclassify(features(itest,:),features(itrain,:),labels(itrain),10,'cosine');
classperf(cp6,knn,itest);

accuracy6=cp6.correctRate;
error6=cp6.errorRate;
sensitivity6=cp6.Sensitivity;
specificity6=cp6.Specificity;