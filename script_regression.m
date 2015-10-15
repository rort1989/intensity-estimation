%% Script for compute regression given data and intensity label for each frame
clear all;
close all;

% load data
src = load('McMaster/McMaster.mat');
% first experiment 
inst{1} = [6]; inst{2} = [5]; inst{3} = [1 4]; inst{4} = [2 6]; inst{5} = [3 5]; inst{6} = [1]; inst{7} = [];
inst{8} = [2]; inst{9} = [8]; inst{10} = [2 5]; inst{11} = []; inst{12} = [3]; inst{13} = [5]; inst{14} = [1]; 
inst{15} = []; inst{16} = [1 9]; inst{17} = []; inst{18} = [4]; inst{19} = []; inst{20} = []; inst{21} = [3]; 
inst{22} = []; inst{23} = []; inst{24} = []; inst{25} = [1]; % ex1

dfactor = 5;
data = cell(1); % features;
id_sub = 0; % id of each sub: can use to find number of seq per sub
intensity = cell(1); % src.PSPI;
labels = cell(1); % begining, apex, end
count_inst = 0;
for s = 1:numel(inst)
    if ~isempty(inst{s})        
        for n = 1:numel(inst{s})
            count_inst = count_inst + 1;
            data{count_inst} = src.PCA_LBP_features{s}{inst{s}(n)}; % features
            intensity{count_inst} = src.PSPI{s}{inst{s}(n)}; % pain intensity: a scalar
            [labels{count_inst}(1,2),labels{count_inst}(1,1)] = min(intensity{count_inst});
            [labels{count_inst}(2,2),labels{count_inst}(2,1)] = max(intensity{count_inst});
            labels{count_inst}(3,1) = numel(intensity{count_inst},1);
            labels{count_inst}(3,2) = intensity{count_inst}(end);
            id_sub(count_inst) = s;
        end        
    end
end

%% feature extraction / dimension reduction / downsampling
% select a subset of features
fdim = size(data,1); % dimension of input features
% checkthe difference of feature values between consecutive frames
fdiff = zeros(1,size(data,2)-1);
for i = 1:length(fdiff)
    fdiff(i) = norm(data(:,i)-data(:,i+1));
end
% % PCA
% D_low = 100;
% [C, eigvalue, proj, mean_train] = myPCA(data, D_low);
% energy = cumsum(eigvalue)./sum(eigvalue);
% plot(energy);

%% define initial parameter of regression model
rng default;
theta0 = randn(fdim+1,1); 
gamma = 1;
% train regression model
% Solving minimization problem using Matlab optimization toolbox
options = optimset('GradObj','on');
% [f0,g0] = regressor(theta0,data,labels);
% numgrad = computeNumericalGradient(@(theta) regressor(theta,data,labels), theta0);
% err = norm(g0-numgrad);
[theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta,data,labels,gamma), theta0, options);

%% test: compute the AU intensity given testing frame and learned model
dec_values =theta'*[data; ones(1,size(data,2))];
RR = corrcoef(dec_values,intensity);  ry = RR(1,2);

%% save results
% save('BP4D/results/model2_3.mat','theta0','theta','f','eflag','output','g','inst','idx_au','dfactor','ry','labels');

%% plot intensity
close all;
for i = 4
    figure;
    T = size(src.AU{i},1);
    idx = 1:dfactor:T;
    plot(1:length(idx),src.AU{i}(idx,1),'r'); hold on; 
    plot(dec_values);
    %axis([0 length(idx) -1 6])
end
