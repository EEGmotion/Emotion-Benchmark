%DEAP subjects list
function output = deap_List

output = {'s01','s02','s03','s04','s05','s06','s07','s08','s09','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21','s22','s23','s24','s25','s26','s27','s28','s29','s30','s31','s32'};

end

list = deap_List;


for s=1:32
    load(['/Users/ginoslanzirodriguez/Desktop/Deap/data_preprocessed_matlab/' list{s}],'data');
    
datos=zeros(8064,40*32);
for i=1:40
for j=1:32
datos(:,j+32*(i-1))=data(i,j,:);
end
end

%save([list{s} '.txt'],'datos','-ascii')

energia=zeros(5,1280);
REE=zeros(3,1280);
LREE=zeros(3,1280);
ALREE=zeros(3,1280);
feats_wavelet=zeros(9,1280);
for i=1:1280
    energia(:,i)=getEnergy(datos(:,i));
    REE(1,i)=energia(1,i)/(energia(1,i)+energia(2,i)+energia(3,i));
    REE(2,i)=energia(2,i)/(energia(1,i)+energia(2,i)+energia(3,i));
    REE(3,i)=energia(3,i)/(energia(1,i)+energia(2,i)+energia(3,i));
    LREE(1,i)=log(REE(1,i));
    LREE(2,i)=log(REE(2,i));
    LREE(3,i)=log(REE(3,i));
    ALREE(1,i)=abs(LREE(1,i));
    ALREE(2,i)=abs(LREE(2,i));
    ALREE(3,i)=abs(LREE(3,i));
    
end

    feats_wavelet(1,:)=REE(1,:);
    feats_wavelet(2,:)=REE(2,:);
    feats_wavelet(3,:)=REE(3,:);
    feats_wavelet(4,:)=LREE(1,:);
    feats_wavelet(5,:)=LREE(2,:);
    feats_wavelet(6,:)=LREE(3,:);
    feats_wavelet(7,:)=ALREE(1,:);
    feats_wavelet(8,:)=ALREE(2,:);
    feats_wavelet(9,:)=ALREE(3,:);
    

    
 save([list{s} '.mat'],'feats_wavelet','datos','REE','LREE','ALREE')   

end

%% store all feats in one mat.
EEG_REE=[];
EEG_LREE=[];
EEG_ALREE=[];

for s=1:32
    load(['/Users/ginoslanzirodriguez/Desktop/gino/WIC/Deap/Features/' list{s}],'REE','LREE','ALREE');
    
    EEG_REE=[EEG_REE;REE];
    EEG_LREE=[EEG_LREE;LREE];
    EEG_ALREE=[EEG_ALREE;ALREE];
    
end

save('eeg_feats.mat','EEG_REE','EEG_LREE','EEG_ALREE')


%create a vector of valence labels.
label_valencia=[];
for i=1:40
    x=kron(label_all(i,1),ones(32,1));
    label_valencia=[label_valencia;x];
end



%% transform claudio's indices:
load('index_final_test_5.mat')
for i=1:256
if rem(index_final_test(i),40) == 0
suj(i)=floor(index_final_test(i)/40);
trial(i)=40;
else
suj(i)=floor(index_final_test(i)/40)+1;
trial(i)=rem(index_final_test(i),40);
end
end
suj=suj';
index_final_test=index_final_test';
trial=trial';
index_final_test=index_final_test';

for i=1:256
if trial(i)==1
indice(i)=suj(i)
else
indice(i)=(trial(i)-1)*32+suj(i);
end
end
indice=indice';

save('indices_test_5.mat','indice')

%% transform claudio's indices:
load('index_final_test_3.mat')
for i=1:256
if rem(index_final_test(i),40) == 0
suj(i)=floor(index_final_test(i)/40);
trial(i)=40;
else
suj(i)=floor(index_final_test(i)/40)+1;
trial(i)=rem(index_final_test(i),40);
end
end
suj=suj';
index_final_test=index_final_test';

for i=1:256
if trial(i)==1
indice(i)=suj(i)
else
indice(i)=(trial(i)-1)*32+suj(i);
end
end
indice=indice';

save('indices_test_3.mat','indice')

