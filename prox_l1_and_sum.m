function op = prox_l1_and_sum( q, b, nColumns, zeroID )

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

% Nov 2017, Stephen.Becker@Colorado.edu

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

% This is Matlab and Octave compatible code
op = tfocs_prox( @(x)f(q,x), @(x,t)prox_f(q,b,nColumns,zeroID,x,t) , 'vector' );
end

% These are now subroutines, that are NOT in the same scope
function v = f(qq,x)
    v = norm( qq(:).*x(:), 1 );
end

function x = prox_f(qq,b,nColumns,zeroID,x,t) % stepsize is t
    tq = t .* qq; % March 2012, allowing vectorized stepsizes
    
    if zeroID
        if nColumns > 1
            X   = reshape( x, [], nColumns );
            for col = 1:nColumns
%                 x   = X(:,col);
%                 x(col) = []; % shorten to a length n-1 x 1 vector
                ind     = [1:col-1,col+1:size(X,1)];
                x       = X(ind,col);
                x       = prox_l1sum( x, tq, b );
                X(col,col)  = 0;
                X(ind,col)  = x;
            end
            x   = X(:);
        else
            x   = prox_l1sum( x, tq, b );
        end    
    else
        if nColumns > 1
            X   = reshape( x, [], nColumns );
            for col = 1:nColumns
                X(:,col) = prox_l1sum( X(:,col), tq, b );
            end
            x   = X(:);
        else
            x   = prox_l1sum( x, tq, b );
        end
    end
end



% Main algorithmic part: if x0 is length n, takes O(n log n) time
function x = prox_l1sum( x0, lambda, b )

    brk_pts = sort( [x0-lambda;x0+lambda], 'descend' );

    shrink  = @(x) sign(x).*max( abs(x) - lambda, 0 );
    xnu     = @(nu) shrink( x0 - nu );
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
        j = round(mean([lwrBnd,uprBnd]));
        printf('j is %d (bounds were [%d,%d])\n', j, lwrBnd,uprBnd );
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
        aBounds = [-Inf,a2];
    elseif uprBnd == length(brk_pts) + 1
        a1 = brk_pts( lwrBnd );
        a2 = a1 + 10; % arbitrary
        aBounds = [a1,Inf];
    else
        % In general case, we can infer linear part from the two break points
        a1 = brk_pts( lwrBnd );
        a2 = brk_pts( uprBnd );
        aBounds = [a1,a2];
    end
    
    % Now we have the support, find exact value
    x       = xnu( mean(aBounds) );  % to find the support
    supp    = find(x);

    sgn     = sign(x);
    nu      = ( sum(x0(supp) - lambda*sgn(supp) ) - b )/length(supp);
    
    x   = xnu( nu );

end
