%% Script for compute regression given data and intensity label for each frame
clear all;
close all;

% load data
src = load('SYN/syn1.mat');

%% feature extraction / dimension reduction
data = src.data;
labels = src.labels;

%% define initial parameter of regression model
rng default;
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
