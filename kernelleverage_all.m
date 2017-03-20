

labeled_indices = initial_labeled_indices;
unlabeled_indices = setdiff((1:size(training_data_scaled,1))', labeled_indices);
if strcmp(kernel_type, 'rbf')
    leverage_values = calculate_leverage(RBF_kernel(get_distance(training_data_scaled),sigma),eig_threshold);
elseif strcmp(kernel_type, 'lin')
    leverage_values = calculate_leverage(linear_kernel(training_data_scaled),eig_threshold);
elseif strcmp(kernel_type, 'poly')
    leverage_values = calculate_leverage(poly_kernel(training_data_scaled,deg,coeff),eig_threshold);
end
for i=1:iter
	tstart = tic;
    query = unlabeled_indices(find(max(leverage_values(unlabeled_indices,:))));
    time_kernelleverage(i,t) = toc(tstart);
    query = query(1);
    labeled_indices=[labeled_indices;query];
    unlabeled_indices = setdiff(unlabeled_indices, query);
    [~,~,accuracy_temp,~,~] = process_svm(training_data_scaled(labeled_indices,:),training_label(labeled_indices),test_data_scaled,test_label,0);
    accuracy_kernelleverage_all(i,t) = accuracy_temp;
end
