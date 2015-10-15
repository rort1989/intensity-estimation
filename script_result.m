%% scripts to generate plot, analyze results
src = load('BP4D/same_exp_dif_sub.mat');
inst = 4; % 14 % select one sequence
idx_au = 1;
dfactor = 5;
% downsampledata(data,intensity,dfactor);
data = src.features{inst}(1:dfactor:end,:)';
data = data(99:end,:); 
intensity = src.AU{inst}(1:dfactor:end,idx_au);
[labels(2),labels(1)] = max(intensity);
xx=load('BP4D/results/model2_3.mat');
dec_values = xx.theta'*[data; ones(1,size(data,2))];
RR = corrcoef(dec_values,intensity);  ry = RR(1,2);

subplot(2,1,1)
plot(intensity); title('intensity level of selected AU')
subplot(2,1,2)
plot(dec_values); title('intensity from learned regression model')
