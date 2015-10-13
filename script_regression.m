%% Script for compute regression given data and intensity label for each frame
clear all;
close all;

% load data
set = 2;
src = load(sprintf('SYN/syn%d.mat',set));

%% feature extraction / dimension reduction
rng default;
p = randperm(src.N);
inst_train = p(1:src.N/2); 
inst_test = p(1+src.N/2:src.N);
data = src.data(inst_train);
labels = src.labels(inst_train);
fdim = src.fdim;

%% define initial parameter of regression model
rng default;
theta0 = randn(fdim+1,1); 
gamma = 1000;
% train regression model
% Solving minimization problem using Matlab optimization toolbox
options = optimset('GradObj','on');
[f0,g0] = regressor(theta0,data,labels,gamma);
numgrad = computeNumericalGradient(@(theta) regressor(theta,data,labels,gamma), theta0);
err = norm(g0-numgrad);
[theta,f,eflag,output,g] = fminunc(@(theta) regressor(theta,data,labels,gamma), theta0, options);%_base

%% test: compute the AU intensity given testing frame and learned model
dec_values = cell(1,length(inst_test));
ry = zeros(1,length(inst_test));
for n = 1:length(inst_test)
    dec_values{n} =theta'*[src.data{inst_test(n)}; ones(1,size(src.data{inst_test(n)},2))];
    RR = corrcoef(dec_values{n},src.intensity{inst_test(n)});  ry(n) = RR(1,2);
end
mean(ry)
%% save results
% save(sprintf('SYN/results/syn%d.mat',set),'theta0','theta','f','eflag','output','g','inst_train','inst_test','ry');

%% plot AU intensity
% close all;
% for n = 1:length(inst_test)
%     figure;
%     T = size(src.data{inst_test(n)},2);
%     plot(src.intensity{inst_test(n)},'r'); hold on; 
%     plot(dec_values{n});
%     axis([0 T -1 src.nstate]);
% end
