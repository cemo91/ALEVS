warning('off','all')
clear
clc
dataset = {'digit1.mat';'g241n.mat';'g241c.mat';'USPS.mat';'ringnorm.mat';'spambase.mat';'MNIST-3vs5.mat';'UvsV.mat';'twonorm.mat'};
names = {'digit1';'g241n';'g241c';'USPS';'ringnorm';'spambase';'MNIST-3vs5';'UvsV';'twonorm'};
random_indices = {'indices-digit1.mat';'indices-g241n.mat';'indices-g241c.mat';'indices-USPS.mat';'indices-ringnorm.mat';'indices-spambase.mat';'indices-MNIST-3vs5.mat';'indices-UvsV.mat';'indices-twonorm.mat'};
name_dir = 'RESULTS';
mkdir(name_dir);
for d=1:size(names,1)
        name = names{d};
        disp(name);
        run('dataset_experimenter');
        cd(name_dir);
        save(name, 'accuracy', 'accuracy_alevs' , 'accuracy_quire', 'accuracy_uncertainty', 'accuracy_random', 'accuracy_kernelleverage_all', 'eigenvalues', 'coherences','queried_leverage', 'queried_indices', 'queried_labels','kernel_sizes','selected_k','queried_predicted_labels', 'time_alevs', 'time_quire', 'time_uncertainty', 'time_random', 'time_kernelleverage');
        cd ..;
end
warning('on','all')
