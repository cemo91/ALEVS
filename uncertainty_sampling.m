function index = uncertainty_sampling(data,label,index_labeled,index_unlabeled)
    [~,~,~,posterior,~] = process_svm(data(index_labeled,:),label(index_labeled),data(index_unlabeled,:),label(index_unlabeled),1);
    confidence = (abs(posterior(:,1) - 0.5) + abs(posterior(:,2) - 0.5))./2;
    index = index_unlabeled(find(confidence == min(confidence),1));
end