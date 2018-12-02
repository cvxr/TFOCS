function [] = test_prox_l1_mat()
    
    check_correctness()
    time_single_run()

end

function [] = check_correctness()
% Check correctness of optimizations done to prox_l1_mat

    % An example from sparse subspace clustering (SSC)
    n_rows = 700;
    n_cols = 600;
    x = randn(n_rows, n_cols);
    x = x(:);
    t = rand();
    t = rand(n_cols,1);
    Q = rand();
    zeroID = true;
    useMex = true;
    
    % Make the prox operators
    prox_ref = prox_l1_mat_ref(Q, n_cols, zeroID);
    prox_test = prox_l1_mat(Q, n_cols, zeroID, useMex);
    
    [~,y_ref] = prox_ref(x, t);
    [~,y_test] = prox_test(x, t);

    %reshape(y_ref, [n_rows n_cols])
    %reshape(y_test, [n_rows n_cols])

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
    useMex = false;
    useMex = true; shrink_mex2(struct('num_threads', 4));
    
    % Make the prox operator
    %prox = prox_l1_mat_ref(Q, n_cols, zeroID);
    prox = prox_l1_mat(Q, n_cols, zeroID, useMex);
    
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

function op = prox_l1_mat_ref( q, nColumns, zeroID)

%PROX_L1_MAT    L1 norm, matricized in a special way
%    OP = PROX_L1_MAT( Q ) implements the nonsmooth function
%        OP(X) = norm(Q.*X,1) with constraints 
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
%
%    OP = PROX_L1_MAT( Q, nColumns )
%       takes the input vector X and reshapes it to have nColumns
%       and applies this prox to every column
%
%    OP = PROX_L1_MAT( Q, nColumns, zeroID )
%       if zeroID == true (it is false by default)
%       then after reshaping X, enforces that X(i,i) = 0
%
% Often useful for sparse subpsace clustering (SSC)
%   See, e.g., https://github.com/stephenbeckr/SSC

% Mar 2018, Stephen.Becker@Colorado.edu

if nargin == 0
    q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) ||  any( q < 0 ) || all(q==0) || numel( q ) ~= 1
    error( 'Argument must be positive.' );
end
if nargin < 2 || isempty( nColumns), nColumns = 1;
else assert( numel(nColumns) == 1 && nColumns >= 1 ); end
if nargin < 3 || isempty( zeroID ), zeroID = false; end

if zeroID && nColumns == 1
    warning('TFOCS:prox_l1_mat:zeroDiag',...
        'You requested enforcing zero diagonals but did not set nColumns>1 which is probably a mistake');
end

% This is Matlab and Octave compatible code
op = tfocs_prox( @(x)f(q,x), @(x,t)prox_f(q,nColumns,zeroID,x,t) , 'vector' );
end

% These are now subroutines, that are NOT in the same scope
function v = f(qq,x)
    v = norm( qq(:).*x(:), 1 );
end

function x = prox_f(qq,nColumns,zeroID,x,t) % stepsize is t
    tq = t .* qq; % March 2012, allowing vectorized stepsizes
    tq = reshape(tq, [1 numel(tq)]);

    % this is fast, but requires more memory
    shrink  = @(x,tq) sign(x).*max( abs(x) - tq, 0 );
    
    if zeroID && nColumns > 1
        X   = reshape( x, [], nColumns );
        n   = size(X,1);
        if nColumns > n
            error('Cannot zero out the diagonal if columns > rows');
        end
        Xsmall = zeros( n-1, nColumns );
        for col = 1:nColumns
            ind     = [1:col-1,col+1:size(X,1)];
            Xsmall(:,col)   = X(ind,col);
        end
        Xsmall = shrink(Xsmall, tq);
        for col = 1:nColumns
            X(col,col)  = 0;
            ind     = [1:col-1,col+1:size(X,1)];
            X(ind,col)  = Xsmall(:,col);
        end
        x   = X(:);
    else
        if nColumns > 1
            X   = reshape( x, [], nColumns );
            X = shrink( X , tq );
            x   = X(:);
        else
            x   = shrink( x , tq);
        end
    end
end


