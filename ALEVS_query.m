

labeled_indices = initial_labeled_indices;
unlabeled_indices = setdiff((1:size(training_data_scaled,1))', labeled_indices);



disp(kernel_type);

for i=1:iter
	tstart = tic;
    [query,queried_leverage(i,t),eigenvalues{i,t,1},eigenvalues{i,t,2},coherences(i,t,1),coherences(i,t,2),selected_k(i,t,1),selected_k(i,t,2),kernel_sizes(i,t,1),kernel_size(i,t,2),queried_predicted_labels(i,t)] = ALEVS(training_data_scaled,training_label,labeled_indices,unlabeled_indices,eig_threshold,kernel_type,sigma,deg,coeff,normalize);
    time_alevs(i,t) = toc(tstart);
    query = query(1);
    queried_indices(i,t) = query;
    labeled_indices=[labeled_indices;query];
    queried_labels(i,t) = training_label(query);
    unlabeled_indices = setdiff(unlabeled_indices, query);
    [~,~,accuracy_temp,~,~] = process_svm(training_data_scaled(labeled_indices,:),training_label(labeled_indices),test_data_scaled,test_label,0);
    accuracy_alevs(i,t) = accuracy_temp;
end
