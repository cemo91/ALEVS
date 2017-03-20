cd('DATASETS');
load(dataset{d});
cd ..;
cd('SPLITS');
load(random_indices{d});
cd ..;
cd('init_idx');
load([names{d} '_init']);
cd ..;
exp_count = 50;
initial_pool = 4;
step_size = 1;
if d==2 || d==3
    iter = 200;
else
    iter = 100;
end

training_data = zeros(ceil(size(X,1)/2),size(X,2));
training_indices = zeros(size(training_data,1),1);
training_label = zeros(size(training_data,1),1);
test_data = zeros((size(X,1)-size(training_data,1)),size(X,2));
sigmas = zeros(iter,exp_count);
%1 pos 2 neg
eigenvalues = cell(iter,exp_count,2);
coherences = zeros(iter,exp_count,2);
kernel_sizes = zeros(iter,exp_count,2);
selected_k = zeros(iter,exp_count,2);
queried_labels = zeros(iter,exp_count);
queried_predicted_labels = zeros(iter,exp_count);
queried_leverage = zeros(iter,exp_count);
queried_indices = zeros(iter,exp_count);
accuracy = zeros(exp_count);
accuracy_alevs = zeros(iter,exp_count);
accuracy_kernelleverage_all = zeros(iter,exp_count);
accuracy_quire = zeros(iter,exp_count);
accuracy_random = zeros(iter,exp_count);
accuracy_uncertainty = zeros(iter,exp_count);

time_alevs = zeros(iter,exp_count);
time_random = zeros(iter,exp_count);
time_uncertainty = zeros(iter,exp_count);
time_quire = zeros(iter,exp_count);
time_kernelleverage = zeros(iter,exp_count);

% initial_indices = cell(exp_count,1);
for t=1:exp_count
    disp(['t=' num2str(t)]);
    r_ind = randomize(:,t);
    training_data = X(r_ind, :);
    training_indices = r_ind;
    training_label = y(r_ind);
    count = 1:size(X,1);
    count = setdiff(count,training_indices);
    test_data = X(count,:);
    test_label = y(count);

    m = mean(training_data);
    v = std(training_data);
    
    training_data_scaled = normalize_data(training_data, m, v);
    test_data_scaled = normalize_data(test_data,m,v);
    
    clear training_data
    clear test_data

%     upper = size(training_data,1);
%     initial_labeled_indices = randperm(upper, initial_pool)';
%     while sum(training_label(initial_labeled_indices)) ~= 0
%         initial_labeled_indices = randperm(upper, initial_pool)';
%     end
%     initial_indices(t) = {initial_labeled_indices};

    initial_labeled_indices = initial_indices{t};
    
    [~,~,accuracy_temp,~,~] = process_svm(training_data_scaled, training_label, test_data_scaled, test_label, 0);
    accuracy(t) = accuracy_temp;

    
    cd('PARAMETERS');
    parameter_name = ['params_' name];
    cd ..;
    load(parameter_name);
    disp('ALEVS');
    run('ALEVS_query');
    
    disp('Leverage on all');
    run('kernelleverage_all');

    disp('random');
    run('random');
    
    disp('uncertainty');
    run('uncertainty');
    
    disp('quire');
    run('quire_query');
end
