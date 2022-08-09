11.3.5 Radial_PSD_updated.m
%radial FFT analysis written by KP Quinn and JX 4-11-11
function [log_radial_freq log_radial_psd] = Radial_PSD_updated(xcol,N,vox_size)
% INPUTS
    % xcol= image input
    % N= image size – i.e. 512x512
    % vox_size = pixels per micron
% OUTPUTS
    % log_radial_psd= radial psd amplitude spectrum (psd y-axis)
    % log_radial_freq= radial psd frequency spectrum (x-axis)
figure (30),imagesc(xcol), hold on;
A= (N/2)-1;
%evaluate 2D FFT
y1 = fft2(double(xcol));
psd = y1.*conj(y1);
psd_n = fftshift(psd); %shift to center for easier radial analysis
S2=log(1+abs(psd_n)); % use abs to compute the magnitude and use log to brighten display
% define (Cartesian) spatial frequency of 2D PSD image
maxfreq = 1/(2*vox_size);
freq=linspace(0,maxfreq,(N/2));
% define radial frequency parameters for annular integration of 2D PSD
delta_freq = maxfreq/A;
freq_annulus = linspace(delta_freq/2,maxfreq-delta_freq/2,A);
radial_freq = zeros(N/2);
annular_psd = linspace(0,0,A);
pixel_count = linspace(0,0,A);
[i,j] = meshgrid(1:N,1:N);
i= i - mean(mean(i));
j= j - mean(mean(j));
[THETA,RHO] = cart2pol(i,j);
rho= round(RHO);
theta= round(THETA);
for x =1:A
    mask= double(rho == x);
    pixel_count(x) = sum(sum(mask));
    temp = psd_n .* mask ;
    inten_count(x)= sum(sum(temp));
end
radial_psd= (inten_count./pixel_count);
% normalize PSD by annuli area to obtain radial PSD
radial_psd_unnormalized=radial_psd(2:A); %ignore first point (high annulus error)
radial_psd_final=radial_psd_unnormalized/sum(radial_psd_unnormalized);
% % send output radial PSD parameters in loglog form
%log_radial_freq = log10(freq_annulus(2:A));
%log_radial_psd = log10(radial_psd_final);
% send output radial PSD parameters in normal form
log_radial_freq = freq_annulus(2:A);
log_radial_psd = radial_psd_final;