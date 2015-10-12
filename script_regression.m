%% Script for compute regression given data and intensity label for each frame
clear all;
close all;

% load data
src = load('BP4D/same_exp_dif_sub.mat');
inst = 4; % 14 % select one sequence
idx_au = 1;
dfactor = 5;
% downsampledata(data,intensity,dfactor);
data = src.features{inst}(1:dfactor:end,:)';
intensity = src.AU{inst}(1:dfactor:end,idx_au);
labels = zeros(3,2); % begining, apex, end
[labels(1,2),labels(1,1)] = min(intensity);
[labels(2,2),labels(2,1)] = max(intensity);
labels(3,1) = size(intensity,1);
labels(3,2) = intensity(end);

%% feature extraction / dimension reduction
% select a subset of features
data = data(99:end,:); % the first 98 are landmark points coordinates
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
theta0 = randn(fdim+1,1); 
theta0 = randn(fdim+1,1);
% train regression model
% Solving minimization problem using Matlab optimization toolbox
options = optimset('GradObj','on');
% [f0,g0] = regressor(theta0,data,labels);
% numgrad = computeNumericalGradient(@(theta) regressor(theta,data,labels), theta0);
% err = norm(g0-numgrad);
[theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta,data,labels), theta0, options);

%% test: compute the AU intensity given testing frame and learned model
dec_values =theta'*[data; ones(1,size(data,2))];
RR = corrcoef(dec_values,intensity);  ry = RR(1,2);

%% save results
% save('BP4D/results/model2_3.mat','theta0','theta','f','eflag','output','g','inst','idx_au','dfactor','ry','labels');

%% plot AU intensity
close all;
for i = 4
    figure;
    T = size(src.AU{i},1);
    idx = 1:dfactor:T;
    plot(1:length(idx),src.AU{i}(idx,1),'r'); hold on; 
%     plot(1:T,AU{i}(:,2),'g'); hold on;
%     plot(1:T,AU{i}(:,3),'b'); hold on;
%     plot(1:T,AU{i}(:,4),'m'); hold on;
%     plot(1:T,AU{i}(:,5),'k'); hold off;    
    plot(dec_values);
    %axis([0 length(idx) -1 6])
end
