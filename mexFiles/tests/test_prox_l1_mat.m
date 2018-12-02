function [] = test_prox_l1_mat()
    
    check_correctness()
    time_single_run()

end

function [] = check_correctness()
% Check correctness of optimizations done to prox_l1_mat

    % An example from sparse subspace clustering (SSC)
    n_rows = 7000;
    n_cols = 6000;
    x = randn(n_rows, n_cols);
    x = x(:);
    t = rand(n_cols,1);
    Q = 1;
    zeroID = true;
    useMatricized = true;
    
    % Make the prox operators
    prox_ref = prox_l1_mat(Q, n_cols, zeroID, useMatricized);
    prox_test = prox_l1_mat_optimized(Q, n_cols, zeroID);
    
    y_ref = prox_ref(x, t);
    y_test = prox_test(x, t);

    rel_err = norm(y_ref - y_test, 'fro') / norm(y_ref, 'fro');
    fprintf('relative error = %1.5e\n', rel_err);

end


function [] = time_single_run()
% Time prox_l1_mat implementations

    minimum_runtime = 1;
    
    % An example from sparse subspace clustering (SSC)
    n_cols = 6000;
    x = randn(n_cols, n_cols);
    x = x(:);
    t = 1;
    Q = 1;
    zeroID = true;
    useMatricized = true;
    
    % Make the prox operator
    %prox = prox_l1_mat(Q, n_cols, zeroID, useMatricized);
    prox = prox_l1_mat_optimized(Q, n_cols, zeroID);
    
    % Warm up
    n_done = 0;
    t_ = tic();
    while true
        if toc(t_) >= minimum_runtime
            break
        end

        y = prox(x, t);
        n_done = n_done + 1;
    end

    % Measure runtime
    times = zeros(n_done,1);
    for n=1:n_done
        t_ = tic();
        y = prox(x, t);
        times(n) = toc(t_);
    end

    fprintf('prox min/mean/max runtime = %1.5e  %1.5e  %1.5e  seconds\n', min(times), mean(times), max(times));

end


