function [index,top_leverage,d1,d2,c_pos,c_neg,k_pos,k_neg,pos_size,neg_size,predicted_label] = ALEVS(data_scaled,label,index_labeled,index_unlabeled,t,type,sigma,degree,coefficient,leverage_normalization)
    [~,prediction,~,~,~] = process_svm(data_scaled(index_labeled,:),label(index_labeled),data_scaled(index_unlabeled,:),label(index_unlabeled),0);
    indices_positive = [index_unlabeled(prediction == 1);index_labeled(label(index_labeled) == 1)];
    indices_negative = [index_unlabeled(prediction == -1);index_labeled(label(index_labeled) == -1)];
    
    if strcmp(type, 'rbf')
        positive_kernel = RBF_kernel(get_distance(data_scaled(indices_positive,:)),sigma);
        negative_kernel = RBF_kernel(get_distance(data_scaled(indices_negative,:)),sigma);
    elseif strcmp(type, 'lin')
        positive_kernel = linear_kernel(data_scaled(indices_positive,:));
        negative_kernel = linear_kernel(data_scaled(indices_negative,:));
    elseif strcmp(type, 'poly')
        positive_kernel = poly_kernel(data_scaled(indices_positive,:),degree,coefficient);
        negative_kernel = poly_kernel(data_scaled(indices_negative,:),degree,coefficient);
    end
    
    pos_size = size(positive_kernel,1);
    neg_size = size(negative_kernel,1);
    
    %k selection in calculate_leverage function
    [positive_leverage,d1,k_pos] = calculate_leverage(positive_kernel,t);
    [negative_leverage,d2,k_neg] = calculate_leverage(negative_kernel,t);
    
    if leverage_normalization == 1
        positive_leverage = positive_leverage.*(pos_size/k_pos);
        negative_leverage = negative_leverage.*(neg_size/k_neg);
    end
    
    c_pos = max(positive_leverage);
    c_neg = max(negative_leverage);
    indices = [index_unlabeled(prediction == 1);index_unlabeled(prediction == -1)];
    leverages = [positive_leverage(1:size(index_unlabeled(prediction == 1),1));negative_leverage(1:size(index_unlabeled(prediction == -1),1))];
    [top_leverage,index]=max(leverages);
    index = indices(index(1));
end