%% Script for compute regression given data and intensity label for each frame
clear all;
close all;

% load data
set = 3;
src = load(sprintf('SYN/syn%d.mat',set));

%% feature extraction / dimension reduction
rng default;
p = randperm(src.N);
inst_train = p(1:src.N/2); 
inst_test = p(1+src.N/2:src.N);
data = src.data(inst_train);
labels = src.labels(inst_train);
fdim = src.fdim;
N = numel(data);
train_data = [];
train_label = [];
for n = 1:N
    nframe = size(labels{n},1);
    for m = 1:nframe
        train_data = [train_data; data{n}(:,labels{n}(m,1))'];
        train_label = [train_label; labels{n}(m,2)];
    end
end
train_data_scaled = train_data;

%% define initial parameter of regression model
svm_param = [3 0 1 1];
configuration = sprintf('-s %d -t %d -g %f -c %f',svm_param(1),svm_param(2),svm_param(3),svm_param(4));
model = svmtrain(train_label, train_data_scaled,configuration);
% get parameter w,b from model
% theta
[predict_label, ~, dec_values_train] = predict(train_label, sparse(train_data_scaled), model);

%% test: compute the AU intensity given testing frame and learned model
dec_values = cell(1,length(inst_test));
ry = zeros(1,length(inst_test));
mse = zeros(1,length(inst_test));
for n = 1:length(inst_test)
    dec_values{n} =theta'*[src.data{inst_test(n)}; ones(1,size(src.data{inst_test(n)},2))]; %
    RR = corrcoef(dec_values{n},src.intensity{inst_test(n)});  ry(n) = RR(1,2);
    e = dec_values{n} - src.intensity{inst_test(n)};
    mse(n) = e(:)'*e(:)/length(e);
end
mean(ry)
mean(mse)
%% save results
% save(sprintf('SYN/results/syn%d.mat',set),'theta0','theta','f','eflag','output','g','inst_train','inst_test','ry','mse');

%% plot AU intensity
close all;
for n = 1%:length(inst_test)
    figure;
    T = size(src.data{inst_test(n)},2);
    plot(src.intensity{inst_test(n)},'r'); hold on; 
    plot(dec_values{n});
    xlabel('frame')
    ylabel('intensity')
    %axis([0 T -1 src.nstate]);
end
