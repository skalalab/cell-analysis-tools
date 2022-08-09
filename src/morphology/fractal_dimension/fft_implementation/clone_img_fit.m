%written by J Xylas
function [roi slope_org slope_cloned log_radial_freq radial_PSD_cloned radial_PSD_org
imFINAL2 PSDvarN PSDvarN_cloned] = clone_img_fit(im, stack2, stack3)
% INPUTS
    % stack2-the beginning of cell layer (first superficial cell layer)
    % stack3- the end of cell layer (last basal cell layer)
% Im - the image to be cloned
% OUTPUTS
    % roi – the cell mask
    % slope_org – slope of the psd of the original images
    % slope_cloned – slope of the psd of the cloned images
    % log_radial_freq – the radial frequency (x-axis)
    % radial_PSD_cloned – the radial PSD of each cloned image (same length as image stack)
    % radial_PSD_org- the radial PSD of each of the original images in the image stack
    % imFINAL2 – the cloned version of the image stack
    % PSDvarN –variance of the PSDs over depth for image stack, normalized to highest freq
    % PSDvarN_cloned –variance of the PSDs over depth for cloned image stack, normalized to
    % highest freq
disp(['starting cloning and fitting Beta Values'])
count=1

for i=stack2:stack3
    im_org=im(:,:,i);
    figure (1), imagesc(im_org), colormap jet;
    roi(:,:,i) = findsignal_v4p2(im_org,15);
    figure (2), imagesc(roi(:,:,i));
    AVal= mean(mean(roi(:,:,i)));
    img_uncorr= im_org.*roi(:,:,i);
    [imFINAL2(:,:,count)]=clonestompzeros(img_uncorr);
    % INPUT image to be cloned
    % OUTPUT is imaged that is cloned.
    [cfun2 gof2 log_radial_freq log_radial_psd2 h2] = PSD_2012_v2(im_org,512, .98);
    [cfun1 gof1 log_radial_freq log_radial_psd1 h1]
    =PSD_2012_v2(imFINAL2(:,:,count),512, .98);
% INPUTS FOR ABOVE
    % img_org = the original image was input
    % imFINAL2-the cloned image was input to PSD
    % N= size of image, i.e. 512
    % percent1= the percent of the PSD that you want to fit (i.e. 98% means you exclude the
    % highest 2% of the PSD – based on amplitude of the PSD
% OUTPUTS
    % cfun – the fitting function
    % gof – the goodness of fit
    % log_radial_freq – the radial frequency (x-axis)
    % log_radial_psd_N - the radial PSD of image
    % h= the frequency point corresponding to the highest frequency fit, based on percent1
radial_PSD_cloned(count, :)= log_radial_psd1;
radial_PSD_org(count, :)= log_radial_psd2;
slope_cloned(count)= cfun1.b
slope_org(count)= cfun2.b
count=count+1
end