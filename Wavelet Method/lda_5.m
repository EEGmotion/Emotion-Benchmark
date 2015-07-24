%% 5 classes

neg1 = label_valencia(:,1)<3.5;
neg2 = label_valencia(:,1) >= 3.5 & label_valencia(:,1) < 4.5;
neu = label_valencia(:,1)< 5.5 & label_valencia(:,1)>=4.5;
pos1 = label_valencia(:,1)>=6.5;
pos2 = label_valencia(:,1) >= 5.5 & label_valencia(:,1) < 6.5;

labels = [];
labels(pos1) = 1;
labels(pos2) = 2;
labels(neu) = 3;
labels(neg1) = 5;
labels(neg2) = 4;
labels = labels';

N = length(labels);

load('indices_test_5.mat')

itest=indice; 
itrain = setdiff(1:N,itest)';

%% LDA REE default prior
features=EEG_REE';

cp1 = classperf(labels);
[ldaClass1,err1,P1,logp1,coeff1] = classify(features(itest,:),features(itrain,:),labels(itrain),'diaglinear');
classperf(cp1,ldaClass1,itest);
accuracy1=cp1.CorrectRate;
error1=cp1.errorRate;
sensitivity1=cp1.Sensitivity;
specificity1=cp1.Specificity;

%% LDA REE prior empirical (indicating that group prior probabilities should be estimated from the group relative frequencies in training). 

cp2 = classperf(labels);
[ldaClass2,err2,P2,logp2,coeff2] = classify(features(itest,:),features(itrain,:),labels(itrain),'diaglinear','empirical');
classperf(cp2,ldaClass2,itest);
accuracy2=cp2.CorrectRate;
error2=cp2.errorRate;
sensitivity2=cp2.Sensitivity;
specificity2=cp2.Specificity;

%% LDA LREE default prior
features=EEG_LREE';

cp3 = classperf(labels);
[ldaClass3,err3,P3,logp3,coeff3] = classify(features(itest,:),features(itrain,:),labels(itrain),'diaglinear');
classperf(cp3,ldaClass3,itest);
accuracy3=cp3.CorrectRate;
error3=cp3.errorRate;
sensitivity3=cp3.Sensitivity;
specificity3=cp3.Specificity;

%% LDA LREE prior empirical (indicating that group prior probabilities should be estimated from the group relative frequencies in training). 

cp4 = classperf(labels);
[ldaClass4,err4,P4,logp4,coeff4] = classify(features(itest,:),features(itrain,:),labels(itrain),'diaglinear','empirical');
classperf(cp4,ldaClass4,itest);
accuracy4=cp4.CorrectRate;
error4=cp4.errorRate;
sensitivity4=cp4.Sensitivity;
specificity4=cp4.Specificity;

%% LDA ALREE default prior
features=EEG_ALREE';

cp5 = classperf(labels);
[ldaClass5,err5,P5,logp5,coeff5] = classify(features(itest,:),features(itrain,:),labels(itrain),'diaglinear');
classperf(cp5,ldaClass5,itest);
accuracy5=cp5.CorrectRate;
error5=cp5.errorRate;
sensitivity5=cp5.Sensitivity;
specificity5=cp5.Specificity;


%% LDA ALREE prior empirical (indicating that group prior probabilities should be estimated from the group relative frequencies in training). 

cp6 = classperf(labels);
[ldaClass6,err6,P6,logp6,coeff6] = classify(features(itest,:),features(itrain,:),labels(itrain),'diaglinear','empirical');
classperf(cp6,ldaClass6,itest);
accuracy6=cp6.CorrectRate;
error6=cp6.errorRate;
sensitivity6=cp6.Sensitivity;
specificity6=cp6.Specificity;