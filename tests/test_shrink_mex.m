function [] = test_shrink_mex()
% Test shrink_mex.c and shrink_mex2.cc

    check_correctness()
    time_single_run()    

end


function [] = check_correctness()
    rng(271828);

    m = 700;
    n = 600;
    X = randn(m,n);
    lambda = rand();
    soffset = randn();
    voffset = randn(1,n);

    shrink_ref = @(X, lambda) sign(X) .* max( abs(X) - lambda, 0);
    shrink_test = @(X, lambda) shrink_mex(X, lambda);
    shrink_mex2(struct('num_threads', 1));
    shrink_test2 = @(X, lambda) shrink_mex2(X, lambda);

    fprintf('Testing shrink(X, lambda):\n');
    ref = shrink_ref(X, lambda);
    test = shrink_test(X, lambda);
    test2 = shrink_test2(X, lambda);
    fprintf('  rel_err  = %1.5e\n', norm(test - ref, 'fro') / norm(ref, 'fro'));
    fprintf('  rel_err2 = %1.5e\n', norm(test2 - ref, 'fro') / norm(ref, 'fro'));

 
    shrink_ref = @(X, lambda, offset) sign(X - offset) .* max( abs(X - offset) - lambda, 0);
    shrink_test = @(X, lambda, offset) shrink_mex(X, lambda, offset);
    shrink_mex2(struct('num_threads', 1));
    shrink_test2 = @(X, lambda, offset) shrink_mex2(X, lambda, offset);


    fprintf('Testing shrink(X, lambda, scalar offset):\n');
    ref = shrink_ref(X, lambda, soffset);
    test = shrink_test(X, lambda, soffset);
    test2 = shrink_test2(X, lambda, soffset);
    fprintf('  rel_err  = %1.5e\n', norm(test - ref, 'fro') / norm(ref, 'fro'));
    fprintf('  rel_err2 = %1.5e\n', norm(test2 - ref, 'fro') / norm(ref, 'fro'));

 
    fprintf('Testing shrink(X, lambda, vector offset):\n');
    ref = shrink_ref(X, lambda, voffset);
    test = shrink_test(X, lambda, voffset);
    test2 = shrink_test2(X, lambda, voffset);
    fprintf('  rel_err  = %1.5e\n', norm(test - ref, 'fro') / norm(ref, 'fro'));
    fprintf('  rel_err2 = %1.5e\n', norm(test2 - ref, 'fro') / norm(ref, 'fro'));

   
end


function [] = time_single_run()
% Time shrinkage implementations

    minimum_runtime = 1;
    
    % An example from sparse subspace clustering (SSC)
    n_rows = 6000;
    n_cols = 6000;
    x = randn(n_rows, n_cols);
    lambda = 1;
    offset = randn(1, n_cols);

    opt = struct('num_threads', 4);
    shrink_mex2(opt);

    % Make the shrinkage operator
    %shrink = @(X, lambda) sign(X) .* max( abs(X) - lambda, 0);
    %shrink = @(X, lambda) shrink_mex(X, lambda);
    shrink = @(X, lambda) shrink_mex2(X, lambda);

    %shrink = @(X, lambda, offset) sign(X - offset) .* max( abs(X - offset) - lambda, 0);
    %shrink = @(X, lambda, offset) shrink_mex(X, lambda, offset);
    %shrink = @(X, lambda, offset) shrink_mex2(X, lambda, offset);

    % Warm up
    n_done = 0;
    t_ = tic();
    while true
        if toc(t_) >= minimum_runtime
            break
        end

        y = shrink(x, lambda);
        %y = shrink(x, lambda, offset);
        n_done = n_done + 1;
    end

    % Measure runtime
    times = zeros(n_done,1);
    for n=1:n_done
        t_ = tic();
        y = shrink(x, lambda);
        %y = shrink(x, lambda, offset);
        times(n) = toc(t_);
    end

    fprintf('shrinkage min/mean/max runtime = %1.5e  %1.5e  %1.5e  seconds\n', min(times), mean(times), max(times));

end
