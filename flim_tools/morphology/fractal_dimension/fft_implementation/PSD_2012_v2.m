% 11.3.4 PSD_2012_v2.m
%This program will take the PSD of an image and fit it
%most updated as of Jan 2012
function [cfun gof log_radial_freq log_radial_psd_N h ]= PSD_2012_v2(im, N, percent1)
% This function calculates the PSD of an image and then fits the PSD three times with random
% initializations of fit parameters to find the fit with the best r-squared value.
% INPUTS
    % im-the image to analyzed for PSD
    % N= size of image, i.e. 512
    % percent1= the percent of the PSD that you want to fit (i.e. 98% means you exclude the
    % highest 2% of the PSD – based on amplitude of the PSD
% OUTPUTS
    % cfun – the fitting function
    % gof – the goodness of fit
    
    % log_radial_freq – the radial frequency (x-axis)
    % log_radial_psd_N - the radial PSD of image
    % h= the frequency point corresponding to the highest frequency fit, based on percent1
    %figure, imagesc(im), colormap gray;
    %take PSD
[log_radial_freq log_radial_psd] = Radial_PSD_updated(im, N , 1/2);
% INPUTS
    % im= image input
    % N= image size – i.e. 512x512
    % vox_size = pixels per micron i.e. 1/2
% OUTPUTS
    % log_radial_psd= radial psd amplitude spectrum (psd y-axis)
    % log_radial_freq= radial psd frequency spectrum (x-axis)
%normalize by mean
m3= mean(log_radial_psd);
log_radial_psd_N= log_radial_psd./m3;
domainMAX = .1;
log_radial_psd_N= smooth(smooth((log_radial_psd_N)));
ffun = fittype('power1');
log_radial_freqROT3= rot90(log_radial_freq);
log_radial_psdROT3= rot90(rot90(log_radial_psd_N));%rot90???
PSDmax = max(log10(log_radial_psdROT3));
PSDmin= min(log10(log_radial_psdROT3));
range1=PSDmax-PSDmin;
percentRange= percent1*range1;
PSDminCalc=PSDmax-percentRange;
newMIN=10^PSDminCalc;
%find where the high frequency values become larger than noise floor
PSDminCalc_x = find(log_radial_psdROT3 > newMIN);
PSDminCalc_xcoor=min(PSDminCalc_x);
h= log_radial_freqROT3(PSDminCalc_xcoor);
outliers = excludedata(log_radial_freqROT3,log_radial_psdROT3,'domain',[domainMAX h]);
%includes only the specified domain
%initalize a random matrix for variables
alpha= rand.*2;
beta= rand.*-5;
% %fit and plot
[cfun1,gof1,output] = fit(log_radial_freqROT3, log_radial_psdROT3, ffun, 'Exclude', outliers,
'Startpoint', [alpha,beta], 'MaxFunEvals', 1e+100, 'MaxIter',1e+100 , 'Robust', 'LAR',
'Lower', [0 -inf],'Upper', [2 1]);
alpha= rand.*2;
beta= rand.*-5;
% %fit and plot
[cfun2,gof2,output] = fit(log_radial_freqROT3, log_radial_psdROT3, ffun, 'Exclude', outliers,
'Startpoint', [alpha,beta], 'MaxFunEvals', 1e+100, 'MaxIter',1e+100 , 'Robust', 'LAR',
'Lower', [0 -inf],'Upper', [2 1]);
alpha= rand.*2;
beta= rand.*-5;
% %fit and plot
[cfun3,gof3,output] = fit(log_radial_freqROT3, log_radial_psdROT3, ffun, 'Exclude', outliers,
'Startpoint', [alpha,beta], 'MaxFunEvals', 1e+100, 'MaxIter',1e+100 , 'Robust', 'LAR',
'Lower', [0 -inf],'Upper', [2 1]);
if gof3.rsquare>=gof2.rsquare && gof3.rsquare>=gof1.rsquare
cfun=cfun3;
gof=gof3;
end
if gof2.rsquare>=gof3.rsquare && gof2.rsquare>=gof1.rsquare
cfun=cfun2;
gof=gof2;
end
if gof1.rsquare>=gof2.rsquare && gof1.rsquare>=gof3.rsquare
cfun=cfun1;
219
gof=gof1;
end
figure(5),loglog(log_radial_freq, log_radial_psd_N), hold on;
figure(5),plot(cfun, 'cyan');
cfun
figure (5), plot(h,newMIN, '*', 'MarkerSize', 20)