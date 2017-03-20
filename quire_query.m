

labeled_indices = initial_labeled_indices;
unlabeled_indices = setdiff((1:size(training_data_scaled,1))', labeled_indices);

for i=1:iter
	tstart = tic;
    query = QUIRE(RBF_kernel(get_distance(training_data_scaled),1),labeled_indices',unlabeled_indices',training_label(labeled_indices),1);
    time_quire(i,t) = toc(tstart);
    labeled_indices=[labeled_indices;query];
    unlabeled_indices = setdiff(unlabeled_indices, query);
    [~,~,accuracy_temp,~,~] = process_svm(training_data_scaled(labeled_indices,:),training_label(labeled_indices),test_data_scaled,test_label,0);
%     accuracy_quire(i,t) = accuracy_temp;
    accuracy_quire(i,t) = accuracy_temp(1);
end
