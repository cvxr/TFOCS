function op = prox_l1_mat_optimized( q, nColumns, zeroID)

%PROX_L1_MAT    L1 norm, matricized in a special way
%    OP = PROX_L1_MAT( Q, nColumns ) implements the nonsmooth function
%        OP(X) = norm(Q.*X,1) with constraints 
%    Q is optional; if omitted, Q=1 is assumed. But if Q is supplied,
%    then it must be a positive real scalar (or must be same size as X).
%
%    OP = PROX_L1_MAT( Q, nColumns )
%       takes the input vector X and reshapes it to size [nRows nColumns]
%       and applies this prox to every column
%
%    OP = PROX_L1_MAT( Q, nColumns, zeroID )
%       if zeroID == true (it is false by default)
%       then after reshaping X, enforces that X(i,i) = 0
%
%
% If available and compiled, this uses TFOCS/mexFiles/shrink_mex2.cc
% to apply the shrinkage operator.  To control the number of threads it uses,
% run
%
%   `shrink_mex2(struct('num_threads', 4))`
%
% when constructing your problem.  This will cache the number of threads internally
% and use that number of threads until you restart MATLAB.
%
%
% Often useful for sparse subpsace clustering (SSC)
%   See, e.g., https://github.com/stephenbeckr/SSC

% Mar 2018, Stephen.Becker@Colorado.edu
% 17 Nov 2018, optimizations by jamesfolberth@gmail.com

if nargin == 0
    q = 1;
elseif ~isnumeric( q ) || ~isreal( q ) ||  any( q < 0 ) || all(q==0) || numel( q ) ~= 1
    error( 'Argument must be positive.' );
end

if nargin < 2 || isempty(nColumns)
    nColumns = 1;
else
    assert( numel(nColumns) == 1 && nColumns >= 1 );
end

if nargin < 3 || isempty( zeroID ), zeroID = false; end

if zeroID && nColumns == 1
    warning('TFOCS:prox_l1_mat:zeroDiag',...
        'You requested enforcing zero diagonals but did not set nColumns>1 which is probably a mistake');
end


% 3/15/18, adding:
%JMF 17 Nov 2018: determine if we have shrink_mex once when constructing the prox handle
if 3~=exist('shrink_mex2','file')
    addpath( fullfile( tfocs_where, 'mexFiles' ) );
end
if 3==exist('shrink_mex2','file') 
    %shrink  = @(x,tq) shrink_mex(x,tq);
    shrink  = @(x,tq) shrink_mex2(x,tq);
else
    % this is fast, but requires more memory
    shrink  = @(x,tq) sign(x).*max( abs(x) - tq, 0 );
end


% This is Matlab and Octave compatible code
op = tfocs_prox( @(x)f(q,x), @(x,t)prox_f(q,nColumns,zeroID,shrink,x,t) , 'vector' );

end

% These are now subroutines, that are NOT in the same scope
function v = f(qq,x)
    v = norm( qq(:).*x(:), 1 );
end

function x = prox_f(qq,nColumns,zeroID,shrink,x,t) % stepsize is t
    tq = t .* qq; % March 2012, allowing vectorized stepsizes
    
    if zeroID && nColumns > 1
        x = reshape( x, [], nColumns );
        nRows = size(x,1);
        if nColumns > nRows
            error('Cannot zero out the diagonal if columns > rows');
        end

        x = shrink(x, tq);

        diag_inds = nRows*(0:nColumns-1) + (1:nColumns);
        x(diag_inds) = 0;

        x = reshape(x, nRows*nColumns, 1);

    else
        if nColumns > 1
            x = reshape( x, [], nColumns );
            nRows = size(x,1);

            x = shrink(x, tq);

            x = reshape(x, nRows*nColumns, 1);
        else
            x   = shrink(x, tq);
        end
    end
end

