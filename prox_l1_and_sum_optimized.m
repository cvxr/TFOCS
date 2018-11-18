function op = prox_l1_and_sum_optimized( q, b, nColumns, zeroID )

%PROX_L1_AND_SUM    L1 norm with sum(X)=b constraints
%    OP = PROX_L1_AND_SUM( Q ) implements the nonsmooth function
%        OP(X) = norm(Q.*X,1) with constraints 
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
%
%    OP = PROX_L1_AND_SUM( Q, B )
%       includes the constraints that sum(X(:)) == B
%       (Default: B=1)
%
%    OP = PROX_L1_AND_SUM( Q, B, nColumns )
%       takes the input vector X and reshapes it to have nColumns
%       and applies this prox to every column
%
%    OP = PROX_L1_AND_SUM( Q, B, nColumns, zeroID )
%       if zeroID == true (it is false by default)
%       then after reshaping X, enforces that X(i,i) = 0
%
%
% If available and compiled, this uses TFOCS/mexFiles/prox_l1_and_sum_worker_mex.cc.
% It also checks for TFOCS/mexFiles/shrink_mex2.cc to use as a slower backup.
% To control the number of threads these use, run
%
%   `prox_l1_and_sum_worker_mex(struct('num_threads', 4))`
%
% when constructing your problem.  This will cache the number of threads internally
% and use that number of threads until you restart MATLAB.
%
% Often useful for sparse subpsace clustering (SSC)
%   See, e.g., https://github.com/stephenbeckr/SSC

% Nov 2017, Stephen.Becker@Colorado.edu
% 18 Nov 2018, optimizations by jamesfolberth@gmail.com

if nargin == 0
    q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) ||  any( q < 0 ) || all(q==0) || numel( q ) ~= 1
    error( 'Argument must be positive.' );
end
if nargin < 2 || isempty(b), b = 1; else, assert( numel(b) == 1 ); end
if nargin < 3 || isempty( nColumns), nColumns = 1;
else assert( numel(nColumns) == 1 && nColumns >= 1 ); end
if nargin < 4 || isempty( zeroID ), zeroID = false; end

if zeroID && nColumns == 1
    warning('TFOCS:prox_l1_and_sum:zeroDiag',...
        'You requested enforcing zero diagonals but did not set nColumns>1 which is probably a mistake');
end


USE_MEX_WORKER = 1;
if USE_MEX_WORKER
    if 3~=exist('prox_l1_and_sum_worker_mex','file')
        addpath( fullfile( tfocs_where, 'mexFiles' ) );
    end
    if 3==exist('prox_l1_and_sum_worker_mex','file')
        op = tfocs_prox( @(x)f(q,x), @(x,t)prox_f_using_worker(q,b,nColumns,zeroID,x,t) , 'vector' );
        return;
    end
    % else fall back to optimized MATLAB approach
end


% 3/15/18, adding:
%JMF 17 Nov 2018: determine if we have shrink_mex once when constructing the prox handle
if 3~=exist('shrink_mex2','file')
    addpath( fullfile( tfocs_where, 'mexFiles' ) );
end
if 3==exist('shrink_mex2','file') 
    %shrink  = @(x,tq) shrink_mex(x,tq);
    %shrink_nu = @(x,tq,nu) shrink_mex(x,tq,nu);
    
    shrink  = @(x,tq) shrink_mex2(x,tq);
    shrink_nu = @(x,tq,nu) shrink_mex2(x,tq,nu);
else
    % this is fast, but requires more memory
    shrink  = @(x,tq) sign(x).*max( abs(x) - tq, 0 );
    shrink_nu = @(x,tq,nu) shrink(x-nu,tq);
end

% This is Matlab and Octave compatible code
op = tfocs_prox( @(x)f(q,x), @(x,t)prox_f(q,b,nColumns,zeroID,shrink,shrink_nu,x,t) , 'vector' );

end

% These are now subroutines, that are NOT in the same scope
function v = f(qq,x)
    v = norm( qq(:).*x(:), 1 );
end

function x = prox_f(qq,b,nColumns,zeroID,shrink,shrink_nu,x,t) % stepsize is t
    tq = t .* qq; % March 2012, allowing vectorized stepsizes
    
    if zeroID && nColumns > 1
        x   = reshape( x, [], nColumns );
        nRows   = size(x,1);
        if nColumns > nRows
            error('Cannot zero out the diagonal if columns > rows');
        end
        
        x = prox_l1sum_zeroID_matricized( x, tq, b, shrink_nu );

        x = reshape(x, nRows*nColumns, 1);

    else
        if nColumns > 1
            x   = reshape( x, [], nColumns );
            nRows   = size(x,1);
            
            x = prox_l1sum_matricized( x, tq, b, shrink_nu );
        
            x = reshape(x, nRows*nColumns, 1);
        else
            x   = prox_l1sum( x, tq, b, shrink_nu );
        end
    end
end


function x = prox_f_using_worker(qq,b,nColumns,zeroID,x,t) % stepsize is t
    tq = t .* qq; % March 2012, allowing vectorized stepsizes
    
    if nColumns > 1 % matrix
        x = reshape(x, [], nColumns);
        nRows = size(x,1);

        if zeroID && nColumns > nRows
            error('Cannot zero out the diagonal if columns > rows');
        end

        x = prox_l1_and_sum_worker_mex(x, tq, b, zeroID);

        x = reshape(x, nRows*nColumns, 1);
    
    else % vector
        x = prox_l1_and_sum_worker_mex(x, tq, b);
    end
end


% Main algorithmic part: if x0 is length n, takes O(n log n) time
function x = prox_l1sum( x0, lambda, b, shrink_nu )

    brk_pts = sort( [x0-lambda;x0+lambda], 'descend' );

    xnu     = @(nu) shrink_nu( x0 , lambda, nu );
    h       = @(x) sum(x) - b; % want to solve h(nu) = 0

    % Bisection
    lwrBnd       = 0;
    uprBnd       = length(brk_pts) + 1;
    iMax         = ceil( log2(length(brk_pts)) ) + 1;
    PRINT = false; % set to "true" for debugging purposes
    if PRINT
        dispp = @disp;
        printf = @fprintf;
    else
        dispp = @(varargin) 1;
        printf = @(varargin) 1;
    end
    dispp(' ');
    for i = 1:iMax
        if uprBnd - lwrBnd <= 1
            dispp('Bounds are too close; breaking');
            break;
        end
        j = round( (lwrBnd+uprBnd)/2 );
        %printf('j is %d (bounds were [%d,%d])\n', j, lwrBnd,uprBnd ); %
        if j==lwrBnd
            dispp('j==lwrBnd, so increasing');
            j = j+1;
        elseif j==uprBnd
            dispp('j==uprBnd, so increasing');
            j = j-1;
        end
        
        a   = brk_pts(j);
        x   = xnu(a);  % the prox
        p   = h(x);
        
        if p > 0
            uprBnd = j;
        elseif p < 0
            lwrBnd = j;
        end
        if PRINT
            % Don't rely on redefinition of printf,
            % since then we would still calculate find(~x)
            % which is slow
            printf('i=%2d, a = %6.3f, p = %8.3f, zeros ', i, a, p );
            if n < 100, printf('%d ', find(~x) ); end
            printf('\n');
        end
    end
    
    % Now, determine linear part, which we infer from two points.
    % If lwr/upr bounds are infinite, we take special care
    % e.g., we make a new "a" slightly lower/bigger, and use this
    % to extract linear part.
    if lwrBnd == 0
        a2 = brk_pts( uprBnd );
        a1 = a2 - 10; % arbitrary
        aBounds = [a1,a2];
    elseif uprBnd == length(brk_pts) + 1
        a1 = brk_pts( lwrBnd );
        a2 = a1 + 10; % arbitrary
        aBounds = [a1,a2];
    else
        % In general case, we can infer linear part from the two break points
        a1 = brk_pts( lwrBnd );
        a2 = brk_pts( uprBnd );
        aBounds = [a1,a2];
    end
    
    % Now we have the support, find exact value
    x       = xnu(( aBounds(1)+aBounds(2))/2 );  % to find the support
    supp    = find(x);

    sgn     = sign(x);
    nu      = ( sum(x0(supp) - lambda*sgn(supp) ) - b )/length(supp);
    
    x   = xnu( nu );

end


% This variant can handle several columns at once,
% and it takes exactly log2(n) iterations, as it doesn't stop early
%   since different columns might stop at different steps and that's
%   not easy to detect efficiently.
function x = prox_l1sum_matricized( x0, lambda, b, shrink_nu )

    brk_pts = sort( [x0-lambda;x0+lambda], 'descend' );
    
    
    xnu     = @(nu) shrink_nu( x0 , lambda, nu );
    
    h       = @(x) sum(x) - b; % want to solve h(nu) = 0

    nCols        = size( x0, 2 ); % allow matrices
    LDA          = size( brk_pts, 1 );
    offsets      = (0:nCols-1)*LDA;%i.e., [0, LDA, 2*LDA, ... ];
    
    lwrBnd       = zeros(1,nCols);
    uprBnd       = (length(brk_pts) + 1)*ones(1,nCols);
    iMax         = ceil( log2(length(brk_pts)) ) + 1;

    for i = 1:iMax

        j = round(mean([lwrBnd;uprBnd]));
        ind = find( j==lwrBnd );
        j( ind ) = j( ind ) + 1;
        ind = find( j==uprBnd );
        j( ind ) = j( ind ) - 1;
        
        a   = brk_pts(j+offsets); % need the offsets to correct it here
        x   = xnu(a);  % the prox
        p   = h(x);
        
        ind = find( p > 0 );
        uprBnd(ind) = j(ind);
        ind = find( p < 0 );
        lwrBnd(ind) = j(ind);

    end

    
    [a1,a2]     = deal( zeros(1,nCols) );
    ind = find( lwrBnd == 0 );
    a2(ind) = brk_pts( uprBnd(ind) + offsets(ind) );
    a1(ind) = a2(ind) - 10;
    ind2 = ind;
    
    ind = find( uprBnd == size(brk_pts,1) + 1 );
    a1(ind) = brk_pts( lwrBnd(ind) + offsets(ind) );
    a2(ind) = a1(ind) + 10;
    
    indOther = setdiff( 1:nCols, [ind2,ind] );
    a1(indOther)    = brk_pts( lwrBnd(indOther) + offsets(indOther) );
    a2(indOther)    = brk_pts( uprBnd(indOther) + offsets(indOther) );
    
    a  = mean( [a1;a2] );
    x       = xnu( a );
    
    nu      = zeros(1,nCols);
    sgn     = sign(x);
    for col = 1:nCols
        supp    = find( sgn(:,col) );
        nu(col)      = ( sum(x0(supp,col) - lambda*sgn(supp,col) ) - b )/length(supp);
    end
    x   = xnu( nu );

end


% This variant can handle several columns at once,
% and it takes exactly log2(n) iterations, as it doesn't stop early
%   since different columns might stop at different steps and that's
%   not easy to detect efficiently.
%
% This hacks "tricks" the implementation into ignoring the diagonal to avoid an extra copy
function x = prox_l1sum_zeroID_matricized( x0, lambda, b, shrink_nu )
    
    [nRows, nCols] = size(x0);
    diag_inds = nRows*(0:nCols-1) + (1:nCols);
    
    % Sort all possible break points
    % We account for ignoring the diagonal by setting the diagonals to -inf,
    % so they're at the bottom of the sorted matrix.  We then need to handle the offsets
    % appropriately.
    
    brk_pts_minus = x0 - lambda; brk_pts_minus(diag_inds) = -inf;
    brk_pts_plus = x0 + lambda; brk_pts_plus(diag_inds) = -inf;
    brk_pts = [brk_pts_minus; brk_pts_plus];
    
    %brk_pts = [x0 - lambda; x0 + lambda];
    %inf_inds = [2*nRows*(0:nCols-1) + (1:nCols),
    %            2*nRows*(0:nCols-1) + (1:nCols) + nRows];
    %brk_pts(inf_inds) = -inf;

    brk_pts = sort( brk_pts, 'descend' );

    
    xnu     = @(nu) shrink_nu( x0, lambda, nu );
    
    h       = @(x) sum(x) - x(diag_inds) - b; % want to solve h(nu) = 0

    LDA          = size( brk_pts, 1 );
    offsets      = (0:nCols-1)*LDA;%i.e., [0, LDA, 2*LDA, ... ];
    num_brk_pts = LDA - 2;
    
    lwrBnd       = zeros(1,nCols);
    uprBnd       = (num_brk_pts + 1 )*ones(1,nCols);
    iMax         = ceil( log2(num_brk_pts) ) + 1;

    for i = 1:iMax

        j = round(mean([lwrBnd;uprBnd]));
        ind = find( j==lwrBnd );
        j( ind ) = j( ind ) + 1;
        ind = find( j==uprBnd );
        j( ind ) = j( ind ) - 1;
        
        a   = brk_pts(j+offsets); % need the offsets to correct it here
        x   = xnu(a);  % the prox
        p   = h(x);
       
        ind = find( p > 0 );
        uprBnd(ind) = j(ind);
        ind = find( p < 0 );
        lwrBnd(ind) = j(ind);

    end
    
    [a1,a2]     = deal( zeros(1,nCols) );
    ind = find( lwrBnd == 0 );
    a2(ind) = brk_pts( uprBnd(ind) + offsets(ind) );
    a1(ind) = a2(ind) - 10;
    ind2 = ind;
    
    ind = find( uprBnd == size(brk_pts,1) + 1 );
    a1(ind) = brk_pts( lwrBnd(ind) + offsets(ind) );
    a2(ind) = a1(ind) + 10;
    
    indOther = setdiff( 1:nCols, [ind2,ind] );
    a1(indOther)    = brk_pts( lwrBnd(indOther) + offsets(indOther) );
    a2(indOther)    = brk_pts( uprBnd(indOther) + offsets(indOther) );
    
    a  = mean( [a1;a2] );
    x       = xnu( a );

    
    sgn     = sign(x);
    supp = (sgn ~= 0); supp(diag_inds) = 0;
    
    UNSAFE_BUT_FAST = true;
    if UNSAFE_BUT_FAST
        % some MATLAB hackery to vectorize the below loop
        % Note that due to roundoff in the first cumsum, this is not necessarily safe
        % With data that are changing signs randomly, the numerical issues should be kept at bay.
        % In initial experiments, this seems to be okay.
        num_supp = sum(supp,1);
        sumthing = cumsum(x0(supp) - lambda*sgn(supp)); % possible numerical issues!
        inds = cumsum(num_supp);
        numerator = sumthing(inds) - b;
        numerator(2:nCols) = numerator(2:nCols) - sumthing(inds(1:nCols-1));
        nu = numerator.' ./ num_supp;
    
    else
        nu      = zeros(1,nCols);
        for col = 1:nCols
            nu(col)      = ( sum(x0(supp(:,col),col) - lambda*sgn(supp(:,col),col) ) - b )/sum(supp(:,col));
        end
    end

    x   = xnu( nu );

    x(diag_inds) = 0;

end

