%% scripts to generate plot, analyze results
% selected sequence for each subject
inst{1} = [6]; % ex1
inst{2} = [5]; % ex1
inst{3} = [1 4]; % ex1
inst{4} = [2 6]; % ex1
inst{5} = [3 5 7]; % ex1 inst 1,2
inst{6} = [1 8]; % ex1
inst{7} = [];
inst{8} = [2 3]; % ex1
inst{9} = [8]; % ex1
inst{10} = [2:7]; % ex1 inst 2,5
inst{11} = [2 5];
inst{12} = [3]; % ex1
inst{13} = [5]; % ex1
inst{14} = [1 2 5]; % ex1 inst 1
inst{15} = [];
inst{16} = [1 5 7 9 11 13 15]; % ex1 inst 1,9
inst{17} = [1];
inst{18} = [1 4]; % ex1 - inst4
inst{19} = [3 4 10];
inst{20} = [2 6];
inst{21} = [1 3 5]; % ex1 - inst3
inst{22} = [1];
inst{23} = [];
inst{24} = [2 6];
inst{25} = [1 5]; % ex1

% plot the pain intensity changes
close all;
src = load('McMaster/McMaster.mat');
for s = 1:numel(src.subject)
    PSPI = src.PSPI{s};
    figure
    for n = 1:numel(inst{s})%numel(PSPI)
        seq = PSPI{inst{s}(n)}; %seq = PSPI{n};
        subplot(1,numel(inst{s}),n);%subplot(2,ceil(numel(PSPI)/2),n);
        plot(seq);
        title(num2str(n));
    end
end

%%
inst_select = [1:4 10 12:21];
for i = 1:length(inst_select)%[1:4 10:12 14:16]%
    subplot(length(inst_select)/3,3,i);
    plot(intensity{inst_select(i)});
    axis([1 length(intensity{inst_select(i)}) 0 9])
end

% subplot(2,1,1)
% plot(intensity); title('intensity level of selected AU')
% subplot(2,1,2)
% plot(dec_values); title('intensity from learned regression model')
