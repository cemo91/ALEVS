function [model, prediction, accuracy, posterior,sigma] = process_svm(data, label, test_data, test_label, posterior_fit)
    if posterior_fit == 0
        model = fitcsvm(data, label, 'KernelFunction', 'RBF');
        prediction = predict(model, test_data);
        accuracy = (sum(prediction == test_label) / size(test_data,1))*100;
        posterior = 0;
        sigma = model.KernelParameters.Scale;
    elseif posterior_fit == 1
        model = fitcsvm(data, label, 'KernelFunction', 'RBF');
        model = fitSVMPosterior(model);
        [prediction, posterior] = predict(model, test_data);
        accuracy = 0;
        sigma = model.KernelParameters.Scale;
    end
end