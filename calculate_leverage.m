function [lev, eig_val,k] = calculate_leverage(matrix,perc)
    [V,D] = eig(matrix);
    eig_val = diag(D);
    [~,idx] = sort(diag(D),1,'descend'); % D nin diagonal elementlerini sort et
    V = V(:, idx); % V yi D deki siraya gore sort et
    
    %k-selection
    k = k_selector(eig_val,perc);

    U1 = V(:,1:k);
    lev = diag(U1*U1');
end