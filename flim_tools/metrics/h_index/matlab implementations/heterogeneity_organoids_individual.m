% Amy Shah

% revised November 25, 2013 - for per-cell xenograft data

% written March 24, 2013

% frequency histograms

clear all; close all; clc;

% read values from excel spreadsheet and graph

% okf6 = xlsread('results_normbyscc', 'okf6');

filename = 'percell_organoids_day1_withperipheral_03292014';

control_in = xlsread(filename, 'control d1');
cetuximab_in = xlsread(filename, 'cetuximab d1');
both_in = xlsread(filename, 'cetuximab cisplatin d1');
cisplatin_in = xlsread(filename, 'cisplatin d1');

% control_ex = xlsread(filename, 'control ex vivo');
% cetuximab_ex = xlsread(filename, 'cetuximab ex vivo');
% bgt226_ex = xlsread(filename, 'BGT226 ex vivo');
% cisplatin_ex = xlsread(filename, 'cisplatin ex vivo');

% write averages and st error to text file

mkdir('organoid_heterogeneity_04202016');
cd('organoid_heterogeneity_04202016')

for i=1:129

control_avg_in = mean(control_in(:,i)); 
control_ste_in = std(control_in(:,i))/sqrt(length(control_in(:,i)));

cetuximab_avg_in = mean(cetuximab_in(:,i));
cetuximab_ste_in = std(cetuximab_in(:,i))/sqrt(length(cetuximab_in(:,i)));

both_avg_in = mean(both_in(:,i));
both_ste_in = std(both_in(:,i))/sqrt(length(both_in(:,i)));

cisplatin_avg_in = mean(cisplatin_in(:,i));
cisplatin_ste_in = std(cisplatin_in(:,i))/sqrt(length(cisplatin_in(:,i)));

% control_avg_ex = mean(control_ex(:,i));
% control_ste_ex = std(control_ex(:,i))/sqrt(length(control_ex(:,i)));
% 
% cetuximab_avg_ex = mean(cetuximab_ex(:,i));
% cetuximab_ste_ex = std(cetuximab_ex(:,i))/sqrt(length(cetuximab_ex(:,i)));
% 
% bgt226_avg_ex = mean(bgt226_ex(:,i));
% bgt226_ste_ex = std(bgt226_ex(:,i))/sqrt(length(bgt226_ex(:,i)));
% 
% cisplatin_avg_ex = mean(cisplatin_ex(:,i));
% cisplatin_ste_ex = std(cisplatin_ex(:,i))/sqrt(length(cisplatin_ex(:,i)));

if i == 6
        title = 'Non-normalized Redox Ratio'; % column G
        value = 1;
elseif i == 39
    title = 'NAD(P)H T1'; % column AN
%     title = 'NAD(P)H \tau_1 (ps)'; % column AN
    value = 1;
elseif i == 50
    title = 'NAD(P)H T2'; % column AY
%     title = 'NAD(P)H \tau_2 (ps)'; % column AY
    value = 1;
elseif i == 72
    title = 'FAD T1'; % column BU
%     title = 'FAD \tau_1 (ps)'; % column BU
    value = 1;
elseif i == 83
    title = 'FAD T2'; % column CF
%     title = 'FAD \tau_2 (ps)'; % column CF
    value = 1;
elseif i == 17
        title = 'NAD(P)H \tau_m (ps)'; % column R
%         title = 'NAD(P)H Tm (ps)';
%  title = 'NADH \tau_m (ps)';
        value = 1;
elseif i == 28
%         title = 'FAD \tau_m (ps)'; % column AC
        title = 'FAD Tm (ps)';
        value = 1;
elseif i == 61
    title = 'Free NAD(P)H (%)'; % column BJ
    value = 1;
% elseif i == 94
%     title = 'FAD \alpha_1'; % column CQ
%     value = 1;
% elseif i == 105
%     title = 'NADH Photons'; % column DB
%     value = 1;
% elseif i == 116
%     title = 'FAD Photons'; %column DM
%     value = 1;
elseif i ==127
    title = 'Redox Ratio'; % column DX
    value = 1;
% elseif i ==127
%     title = 'OMI index = redox ratio/<redox ratio> - NADH a1/<NADH a1> + FAD a2/<FAD a2>'; % column DX
%     value = 1;
elseif i ==128
    title = 'Free FAD (%)'; % column DY
    value = 1;
% elseif i ==129
%     title = 'OMI Index'; % column DZ OMI index = redox ratio/<redox ratio>+NADH a1/<NADH a1> + FAD a2/<FAD a2>
%     value = 1;
else
    title = '';
    value = 0;
end

if value == 1

%     % calculate frequency histograms
%     [control_elements,control_centers] = hist(control_in(:,i),sqrt(length(control_in(:,i))));
%     [cetuximab_elements,cetuximab_centers] = hist(cetuximab_in(:,i),sqrt(length(cetuximab_in(:,i))));
%     [bgt226_elements,bgt226_centers] = hist(bgt226_in(:,i),sqrt(length(bgt226_in(:,i))));
%     [cisplatin_elements,cisplatin_centers] = hist(cisplatin_in(:,i),sqrt(length(cisplatin_in(:,i))));
%     
%     % plot frequency histograms (population density histograms)
% figure(i)
% plot(control_centers, control_elements./max(control_elements), 'b')
% hold on
% plot(cetuximab_centers, cetuximab_elements./max(cetuximab_elements), 'r')
% plot(bgt226_centers, bgt226_elements./max(bgt226_elements), 'g')
% plot(cisplatin_centers, cisplatin_elements./max(cisplatin_elements), 'm')
% hold off
% legend('Control','Cetuximab','BGT226','Cisplatin',0)
% xlabel(title)
% ylabel('Normalized Number of Cells')

% fit to gaussian curve
% fit(cetuximab_centers', cetuximab_elements', 'gauss1')

% from Alex to fit to 1, 2, or 3- component gaussian curves
control_test1 = gmdistribution.fit([control_in(:,i)],1);
control_test2 = gmdistribution.fit([control_in(:,i)],2);
control_test3 = gmdistribution.fit([control_in(:,i)],3);

cetuximab_test1 = gmdistribution.fit([cetuximab_in(:,i)],1);
cetuximab_test2 = gmdistribution.fit([cetuximab_in(:,i)],2);
cetuximab_test3 = gmdistribution.fit([cetuximab_in(:,i)],3);

both_test1 = gmdistribution.fit([both_in(:,i)],1);
both_test2 = gmdistribution.fit([both_in(:,i)],2);
both_test3 = gmdistribution.fit([both_in(:,i)],3);

cisplatin_test1 = gmdistribution.fit([cisplatin_in(:,i)],1);
cisplatin_test2 = gmdistribution.fit([cisplatin_in(:,i)],2);
cisplatin_test3 = gmdistribution.fit([cisplatin_in(:,i)],3);

    [control_elements,control_centers] = hist(control_in(:,i),sqrt(length(control_in(:,i))));
    [cetuximab_elements,cetuximab_centers] = hist(cetuximab_in(:,i),sqrt(length(cetuximab_in(:,i))));
    [both_elements,both_centers] = hist(both_in(:,i),sqrt(length(both_in(:,i))));
    [cisplatin_elements,cisplatin_centers] = hist(cisplatin_in(:,i),sqrt(length(cisplatin_in(:,i))));
    
    % plot graphs
    
figure(i)
hold on

% for the legend
 ControlGroup = hggroup;
    CetuximabGroup = hggroup;
    CisplatinGroup = hggroup;
    BothGroup = hggroup;

% for jj = 1:4
    for jj = 4:4
%     for jj = 1:3 % omit combination treatment
%     for jj = 1:2
    
% subplot(2,2,jj)
if jj == 1
    centers = control_centers;
    data = control_in(:,i);
    label = 'Control';
%     color = 'k'; % black for control
    color = 'b'; % blue for control; gets overwritte
    group = ControlGroup;
elseif jj == 2
    centers = cetuximab_centers;
    data = cetuximab_in(:,i);
    label = 'Cetuximab';
    color = 'r'; % red for cetuximab; gets overwritten
    group = CetuximabGroup;
elseif jj == 3
    centers = cisplatin_centers;
    data = cisplatin_in(:,i);
    label = 'Cisplatin';
%     color = 'b'; % blue for cisplatin
color = 'g'; % green for cisplatin; gets overwritten
    group = CisplatinGroup;
elseif jj == 4
    centers = both_centers;
    data = both_in(:,i);
%     label = 'Cetuximab + Cisplatin';
label = 'Combination';
%     color = 'g'; % green for cetuximab + cisplatin
    color = 'r'; % red for cetuximab + cisplatin; gets overwritten
    group = BothGroup;
end

step_size = (max(centers)-min(centers))/30;
% v = min(centers):step_size:max(centers);
% v = 0:0.1:3;
v = 600:20:2000;
% v = 0:step_size:1.1*max(centers); % for redox ratio, t1, fad tm
% v = 0.9*min(centers):step_size:1.1*max(centers); % for most parameters
% v = 0.95*min(centers):step_size:1*max(centers);
% v = 1*min(bgt226_centers):step_size:1.1*max(cetuximab_centers);
n = hist(data,v);

% need to automate choosing min number of fit parameters
data_test1 = gmdistribution.fit(data,1);
data_test2 = gmdistribution.fit(data,2);
data_test3 = gmdistribution.fit(data,3);

AICvector = [1 2 3; data_test1.AIC data_test2.AIC, data_test3.AIC];
for kk = 1:3
%     if min(AICvector(2,:)) == AICvector(2,kk);
%         min_AIC = kk;
if jj==1 %% new part 05202016
    min_AIC = 1;
elseif jj == 2
    min_AIC = 2;
elseif jj == 3
    min_AIC = 2;
elseif jj ==4
    min_AIC = 1;
    end
end

if min_AIC == 1
    gmfit = gmdistribution.fit(data,min_AIC);
y1 = normpdf(v, gmfit.mu(1,1),(gmfit.Sigma(:,:,1))^(1/2));
y1 = gmfit.PComponents(1,1)*y1; %PComponents = mixing proportions
y = y1;
y = trapz(v,n)*y;

% mean_vector = gmfit.mu;
% weight_vector = gmfit.PComponents;
% sigma_vector = gmfit.Sigma;
% shannon = -1.*[weight_vector(1).*log(weight_vector(1))];
% simpson = weight_vector(1)^2;
% mean_weight_stdev = mean_vector(1).*weight_vector(1).*sigma_vector(1);
% mean_weight = mean_vector(1).*weight_vector(1);
% shannon_mediandistance = -1.*[abs(mean_vector(1)-median(data)).*weight_vector(1).*log(weight_vector(1))];
% weight_mediandistance = weight_vector(1).*abs(mean_vector(1)-median(data));
% shannon_mediandistance_offset = -1.*[abs(mean_vector(1)-median(data(2:end))).*weight_vector(1).*log(weight_vector(1))+sigma_vector(1)];

% title_txt = sprintf('%s %s %s', title, label, '.txt');
% file = fopen(title_txt, 'w');
%     fprintf(file, '%s\t%f\r\n', 'mean' , mean_vector(1));
%     fprintf(file, '%s\t%f\r\n', 'weight' , weight_vector(1));
%     fprintf(file, '%s\t%f\r\n', 'stdev' , sigma_vector(1));
%     fprintf(file, '%s\t%f\r\n', 'shannon' , shannon);
%     fprintf(file, '%s\t%f\r\n', 'simpson' , simpson);
%     fprintf(file, '%s\t%f\r\n', 'mean_weight_stdev' , mean_weight_stdev);
%     fprintf(file, '%s\t%f\r\n', 'mean_weight' , mean_weight);
%     fprintf(file, '%s\t%f\r\n', 'shannon_mediandistance' , shannon_mediandistance);
%     fprintf(file, '%s\t%f\r\n', 'weight_mediandistance' , weight_mediandistance);
%     fprintf(file, '%s\t%f\r\n', 'shannon_mediandistance_offset' , shannon_mediandistance_offset);
% fclose(file);

% bar(v,n)
% hold on
% plot(v,y,'r')
% plot(v, y1*trapz(v,n),'g')
% hold off
% xlabel(title)
% ylabel(axis_label)
curve1 = plot(v,y1,'k', 'LineWidth',8); % plot individual curves
% curve1 = plot(v,y1,color, 'LineWidth',8); % plot combined curves
% curve1 = plot(v,y1,color, 'LineWidth',8); % plot combined curves
% if jj==3
%     set(curve1, 'Color', [130/255,4/255,162/255])
% end

% if jj==1
%     set(curve1, 'Color', [57/255, 106/255, 177/255]) %blue
% end
% set(curve1, 'Parent', group);
% if jj==2
%     set(curve1, 'Color', [218/255, 124/255, 48/255]) %orange
% end
% set(curve1, 'Parent', group);
% if jj==3
%     set(curve1, 'Color', [62/255, 150/255, 81/255]) %green
% end
% set(curve1, 'Parent', group);
% if jj==4
%     set(curve1, 'Color', [146/255, 36/255, 40/255]) %brown
% end
% set(curve1, 'Parent', group);

elseif min_AIC == 2
    gmfit = gmdistribution.fit(data,min_AIC);
y1 = normpdf(v, gmfit.mu(1,1),(gmfit.Sigma(:,:,1))^(1/2));
y2 = normpdf(v, gmfit.mu(2,1),(gmfit.Sigma(:,:,2))^(1/2));
y1 = gmfit.PComponents(1,1)*y1;
y2 = gmfit.PComponents(1,2)*y2;
y = y1+y2;
y = trapz(v,n)*y;

% mean_vector = gmfit.mu;
% weight_vector = gmfit.PComponents;
% sigma_vector = gmfit.Sigma;
% shannon = -1.*[weight_vector(1).*log(weight_vector(1))+weight_vector(2).*log(weight_vector(2))];
% simpson = weight_vector(1)^2+weight_vector(2)^2;
% mean_weight_stdev = mean_vector(1).*weight_vector(1).*sigma_vector(1)+mean_vector(2).*weight_vector(2).*sigma_vector(2);
% mean_weight = mean_vector(1).*weight_vector(1)+mean_vector(2).*weight_vector(2);
% shannon_mediandistance = -1.*[abs(mean_vector(1)-median(data)).*weight_vector(1).*log(weight_vector(1))+abs(mean_vector(2)-median(data)).*weight_vector(2).*log(weight_vector(2))];
% weight_mediandistance = weight_vector(1).*abs(mean_vector(1)-median(data))+weight_vector(2).*abs(mean_vector(2)-median(data));
% shannon_mediandistance_offset = -1.*[abs(mean_vector(1)-median(data(2:end))).*weight_vector(1).*log(weight_vector(1))+sigma_vector(1)+abs(mean_vector(2)-median(data(2:end))).*weight_vector(2).*log(weight_vector(2))+sigma_vector(2)];

% title_txt = sprintf('%s %s %s', title, label, '.txt');
% file = fopen(title_txt, 'w');
%     fprintf(file, '%s\t%f\t%f\r\n', 'mean' , mean_vector(1), mean_vector(2));
%     fprintf(file, '%s\t%f\t%f\r\n', 'weight' , weight_vector(1), weight_vector(2));
%     fprintf(file, '%s\t%f\t%f\r\n', 'stdev' , sigma_vector(1), sigma_vector(2));
%     fprintf(file, '%s\t%f\r\n', 'shannon' , shannon);
%     fprintf(file, '%s\t%f\r\n', 'simpson' , simpson);
%     fprintf(file, '%s\t%f\r\n', 'mean_weight_stdev' , mean_weight_stdev);
%     fprintf(file, '%s\t%f\r\n', 'mean_weight' , mean_weight);
%     fprintf(file, '%s\t%f\r\n', 'shannon_mediandistance' , shannon_mediandistance);
%     fprintf(file, '%s\t%f\r\n', 'weight_mediandistance' , weight_mediandistance);
%     fprintf(file, '%s\t%f\r\n', 'shannon_mediandistance_offset' , shannon_mediandistance_offset);
% fclose(file);

% bar(v,n)
% hold on
% plot(v,y,'r')
% plot(v, y1*trapz(v,n),'g')
% plot(v, y2*trapz(v,n),'g')
% hold off
% xlabel(title)
% ylabel(axis_label)
% curve1 = plot(v,y1,'k', 'LineWidth',8); % plot individual curves
% curve2 = plot(v,y2,'k', 'LineWidth',8);
% set(curve1, 'Parent', group);
% set(curve2, 'Parent', group);
% if jj==3
%     set(curve1, 'Color', [130/255,4/255,162/255])
%     set(curve2, 'Color', [130/255,4/255,162/255])
% end
% curve1 = plot(v,y1+y2,color, 'LineWidth',4); % plot combined curves
curve1 = plot(v,y1+y2,color, 'LineWidth',8); % plot combined curves
% if jj==3
%     set(curve1, 'Color', [130/255,4/255,162/255])
% end
% if jj==1
%     set(curve1, 'Color', [57/255, 106/255, 177/255]) %blue
% end
% set(curve1, 'Parent', group);
% if jj==2
%     set(curve1, 'Color', [218/255, 124/255, 48/255]) %orange
% end
% set(curve1, 'Parent', group);
% if jj==3
%     set(curve1, 'Color', [62/255, 150/255, 81/255]) %green
% end
% set(curve1, 'Parent', group);
% if jj==4
%     set(curve1, 'Color', [146/255, 36/255, 40/255]) %brown
% end
% set(curve1, 'Parent', group);
    
elseif min_AIC == 3
   gmfit = gmdistribution.fit(data,min_AIC);
y1 = normpdf(v, gmfit.mu(1,1),(gmfit.Sigma(:,:,1))^(1/2));
y2 = normpdf(v, gmfit.mu(2,1),(gmfit.Sigma(:,:,2))^(1/2));
y3 = normpdf(v, gmfit.mu(3,1),(gmfit.Sigma(:,:,3))^(1/2));
y1 = gmfit.PComponents(1,1)*y1;
y2 = gmfit.PComponents(1,2)*y2;
y3 = gmfit.PComponents(1,3)*y3;
y = y1+y2+y3;
y = trapz(v,n)*y;

% mean_vector = gmfit.mu;
% weight_vector = gmfit.PComponents;
% sigma_vector = gmfit.Sigma;
% shannon = -1.*[weight_vector(1).*log(weight_vector(1))+weight_vector(2).*log(weight_vector(2))+weight_vector(3).*log(weight_vector(3))];
% simpson = weight_vector(1)^2+weight_vector(2)^2+weight_vector(3)^2;
% mean_weight_stdev = mean_vector(1).*weight_vector(1).*sigma_vector(1)+mean_vector(2).*weight_vector(2).*sigma_vector(2)+mean_vector(3).*weight_vector(3).*sigma_vector(3);
% mean_weight = mean_vector(1).*weight_vector(1)+mean_vector(2).*weight_vector(2)+mean_vector(3).*weight_vector(3);
% shannon_mediandistance = -1.*[abs(mean_vector(1)-median(data)).*weight_vector(1).*log(weight_vector(1))+abs(mean_vector(2)-median(data)).*weight_vector(2).*log(weight_vector(2))+abs(mean_vector(3)-median(data)).*weight_vector(3).*log(weight_vector(3))];
% weight_mediandistance = weight_vector(1).*abs(mean_vector(1)-median(data))+weight_vector(2).*abs(mean_vector(2)-median(data))+weight_vector(3).*abs(mean_vector(3)-median(data));
% shannon_mediandistance_offset = -1.*[abs(mean_vector(1)-median(data(2:end))).*weight_vector(1).*log(weight_vector(1))+sigma_vector(1)+abs(mean_vector(2)-median(data(2:end))).*weight_vector(2).*log(weight_vector(2))+sigma_vector(2)+abs(mean_vector(3)-median(data(2:end))).*weight_vector(3).*log(weight_vector(3))+sigma_vector(3)];

% title_txt = sprintf('%s %s %s', title, label, '.txt');
% file = fopen(title_txt, 'w');
%     fprintf(file, '%s\t%f\t%f\t%f\r\n', 'mean' , mean_vector(1), mean_vector(2), mean_vector(3));
%     fprintf(file, '%s\t%f\t%f\t%f\r\n', 'weight' , weight_vector(1), weight_vector(2), weight_vector(3));
%     fprintf(file, '%s\t%f\t%f\t%f\r\n', 'stdev' , sigma_vector(1), sigma_vector(2), sigma_vector(3));
%     fprintf(file, '%s\t%f\r\n', 'shannon' , shannon);
%     fprintf(file, '%s\t%f\r\n', 'simpson' , simpson);
%     fprintf(file, '%s\t%f\r\n', 'mean_weight_stdev' , mean_weight_stdev);
%     fprintf(file, '%s\t%f\r\n', 'mean_weight' , mean_weight);
%     fprintf(file, '%s\t%f\r\n', 'shannon_mediandistance' , shannon_mediandistance);
%     fprintf(file, '%s\t%f\r\n', 'weight_mediandistance' , weight_mediandistance);
%     fprintf(file, '%s\t%f\r\n', 'shannon_mediandistance_offset' , shannon_mediandistance_offset);
% fclose(file);

% bar(v,n)
% hold on
% plot(v,y,'r')
% plot(v, y1*trapz(v,n),'g')
% plot(v, y2*trapz(v,n),'g')
% plot(v, y3*trapz(v,n),'g')
% hold off
% xlabel(title)
% ylabel(axis_label)
curve1 = plot(v,y1,'k', 'LineWidth',8); % plot individual curves
curve2 = plot(v,y2,'k', 'LineWidth',8);
curve3 = plot(v,y3,'k', 'LineWidth',8); 
% curve3 = plot(v,y3.*100,'k', 'LineWidth',8); % multiply by 100
% set(curve1, 'Parent', group);
% set(curve2, 'Parent', group);
% set(curve3, 'Parent', group);
% if jj==3
%     set(curve1, 'Color', [130/255,4/255,162/255])
%     set(curve2, 'Color', [130/255,4/255,162/255])
%     set(curve3, 'Color', [130/255,4/255,162/255])
% end
% curve1 = plot(v,y1+y2+y3,color, 'LineWidth',4); % plot combined curves
% curve1 = plot(v,y1+y2+y3,color, 'LineWidth',8); % plot combined curves
% if jj==3
%     set(curve1, 'Color', [130/255,4/255,162/255])
% end
% if jj==1
%     set(curve1, 'Color', [57/255, 106/255, 177/255]) %blue
% end
% set(curve1, 'Parent', group);
% if jj==2
%     set(curve1, 'Color', [218/255, 124/255, 48/255]) %orange
% end
% set(curve1, 'Parent', group);
% if jj==3
%     set(curve1, 'Color', [62/255, 150/255, 81/255]) %green
% end
% set(curve1, 'Parent', group);
% if jj==4
%     set(curve1, 'Color', [146/255, 36/255, 40/255]) %brown
% end
% set(curve1, 'Parent', group);


% if jj==1 % black for control
%     xlabel = title;
%     ylabel = 'Normalized Number of Cells';
%     plot(v,y1,'k')
%     plot(v,y2,'k')
%     plot(v,y3,'k')
%     hold on
% elseif jj==2 % blue for cetuximab
%     plot(v,y1,'b')
%     plot(v,y2,'b')
%     plot(v,y3,'b')
%     hold on
%     elseif jj==3 % green for BGT226
%     plot(v,y1,'g')
%     plot(v,y2,'g')
%     plot(v,y3,'g')
%     hold on
%     elseif jj==4 % red for cisplatin
%     plot(v,y1,'r')
%     plot(v,y2,'r')
%     plot(v,y3,'r')
%     legend('Control','Cetuximab','BGT226','Cisplatin',0);
%     hold off
% end

% 219-295

 end % if statement   
 
 
 end % for loop for plotting
 set(get(get(ControlGroup, 'Annotation'),'LegendInformation'),...
     'IconDisplayStyle','on');
  set(get(get(CetuximabGroup, 'Annotation'),'LegendInformation'),...
     'IconDisplayStyle','on');
 set(get(get(CisplatinGroup, 'Annotation'),'LegendInformation'),...
     'IconDisplayStyle','on'); 
 set(get(get(BothGroup, 'Annotation'),'LegendInformation'),...
     'IconDisplayStyle','on');
  
 
%  legend('Control', 'Cetuximab','Cisplatin','Cetuximab+Cisplatin',0, 'FontSize',30);
%  legend('Control', 'Cetuximab','Cisplatin','Combination',0, 'FontSize',45);
%  legend('Control', 'Cetuximab',0, 'FontSize',18);

% xlabel(title, 'FontSize',40);
% ylabel('Normalized Number of Cells','FontSize',40);
xlabel(title, 'FontSize',58); %change from 42 to 50
ylabel({'Normalized'; 'Number of Cells'},'FontSize',58); %change from 42 to 50
set(gca,'FontSize',40);
 hold off
 
% h = gcf;
% title_txt = sprintf('%s %s', title, '.bmp');
% saveas(h,title_txt);

% mkdir('xenografts_AIC');
% cd('xenografts_AIC')
% title_txt = sprintf('%s %s %s', title, 'AIC', '.txt');
% file = fopen(title_txt, 'w');
% 
% headings = char('control test 1', 'control test 2', 'control test 3', 'cetuximab test 1', 'cetuximab test 2', 'cetuximab test 3', 'cetuximab+cisplatin test 1', 'cetuximab+cisplatin test 2', 'cetuximab+cisplatin test 3', 'cisplatin test 1', 'cisplatin test 2', 'cisplatin test 3');
% AIC = [control_test1.AIC control_test2.AIC control_test3.AIC cetuximab_test1.AIC cetuximab_test2.AIC cetuximab_test3.AIC both_test1.AIC both_test2.AIC both_test3.AIC cisplatin_test1.AIC cisplatin_test2.AIC cisplatin_test3.AIC];
% 
% for j=1:length(AIC)
%     fprintf(file, '%s\t%f\r\n', headings(j,:), AIC(j));
% end
%fclose(file);

 else
end

end

cd ..