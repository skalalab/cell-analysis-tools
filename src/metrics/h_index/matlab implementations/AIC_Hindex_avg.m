clear subpops
% clear HindexWV
% clear HindexD2
% clear Hindex3
% clear HindexP1
% clear HindexP2
% clear HindexP
% clear minimumP
% clear HindexSD

for a=1:100
    clc;
    a
fit1  = gmdistribution.fit(data,1);
fit2  = gmdistribution.fit(data,2);
fit3  = gmdistribution.fit(data,3);

AICvector = [1 2 3; fit1.AIC fit2.AIC, fit3.AIC];
for kk = 1:3
    if min(AICvector(2,:)) == AICvector(2,kk);
        min_AIC = kk;
    end
end

min_AIC = 2; % Ignores best fit and forces to desired # of components
subpops(a)=min_AIC;

if min_AIC == 1
    
    HindexWV(a) = fit1.Sigma(:,:);
%     HindexSD(a) = fit1.Sigma(:,:).^0.5;
%     HindexD2(a) = fit1.Sigma(:,:);
%     Hindex3(a)  =(fit1.Sigma.^0.5)/3;
%     HindexP1(a) = (fit1.Sigma(:,:).^0.5).*(1-log(2));
%     HindexP2(a) = fit1.Sigma(:,:);
%     HindexP(a) = fit1.Sigma(:,:);
%     minimumP(a) = 1;

elseif min_AIC == 2
        
    HindexWV(a) = sum(((fit2.Sigma(:,:)').*fit2.PComponents')     -(abs((fit2.mu-median(data))).*fit2.PComponents'.*log(fit2.PComponents')));
%     HindexSD(a) = sum(((fit2.Sigma(:,:)'.^0.5).*fit2.PComponents')-(abs((fit2.mu-median(data))).*fit2.PComponents'.*log(fit2.PComponents')));
%     HindexD2(a) = sum(((fit2.Sigma(:,:)').*fit2.PComponents')     -(((fit2.mu-median(data)).^2).*fit2.PComponents'.*log(fit2.PComponents')));
%     Hindex3(a)  = sum((((fit2.Sigma(:,:)'.^0.5)/3).*fit2.PComponents')     - (abs(fit2.mu-median(data))    .*fit2.PComponents'.*log(fit2.PComponents')));
%     HindexP1(a) = sum((fit2.Sigma(:,:)'.^0.5 + abs(fit2.mu-median(data))).*(1-(fit2.PComponents'.*log(fit2.PComponents'+1))));
%     HindexP2(a) = sum((fit2.Sigma(:,:)'.^0.5 + abs(fit2.mu-median(data))).*(fit2.PComponents'.^2));
%     HindexP(a)  = sum((fit2.Sigma(:,:)'.^0.5 + abs(fit2.mu-median(data))).*fit2.PComponents');
%     minimumP(a) = min(fit2.PComponents);
    
elseif min_AIC == 3
        
    HindexWV(a) = sum(((fit3.Sigma(:,:)').*fit3.PComponents')     -(abs((fit3.mu-median(data))).*fit3.PComponents'.*log(fit3.PComponents')));
%     HindexSD(a) = sum(((fit3.Sigma(:,:)'.^0.5).*fit3.PComponents')-(abs((fit3.mu-median(data))).*fit3.PComponents'.*log(fit3.PComponents')));
%     HindexD2(a) = sum(((fit3.Sigma(:,:)').*fit3.PComponents')     -(((fit3.mu-median(data)).^2).*fit3.PComponents'.*log(fit3.PComponents')));
%     Hindex3(a)  = sum((((fit3.Sigma(:,:)'.^0.5)/3).*fit3.PComponents') -(abs(fit3.mu-median(data))    .*fit3.PComponents'.*log(fit3.PComponents')));
%     HindexP1(a) = sum((fit3.Sigma(:,:)'.^0.5 + abs(fit3.mu-median(data))).*(1-(fit3.PComponents'.*log(fit3.PComponents'+1))));
%     HindexP2(a) = sum((fit3.Sigma(:,:)'.^0.5 + abs(fit3.mu-median(data))).*(fit3.PComponents'.^2));
%     HindexP(a)  = sum((fit3.Sigma(:,:)'.^0.5 + abs(fit3.mu-median(data))).*fit3.PComponents');
%     minimumP(a) = min(fit3.PComponents);
end
end
%H=mean(Hindex)
HWV = mean(HindexWV)
HWVstd = std(HindexWV)
% HSD = mean(HindexSD)
% HSDsd = std(HindexSD)
% HD2 = mean(HindexD2)
% HD2sd = std(HindexD2)
% H3 = mean(Hindex3)
% HP1 = mean(HindexP1)
% HP1sd = std(HindexP1)
% HP2 = mean(HindexP2)
% HP = mean(HindexP)
% H3sd = std(Hindex3)
% mean(minimumP)
 S=mean(subpops)
% 
