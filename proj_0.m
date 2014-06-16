function op = proj_0(offset,A,algo)
%PROJ_0     Projection onto the set {0}
%    OP = PROJ_0 returns an implementation of the indicator 
%    function for the set including only zero.
%
%    OP = PROJ_0( c ) returns an implementation of the
%    indicator function of the set {c}
%    If c is a scalar, this is interpreted as c*1
%    where "1" is the all ones object of the appropriate size.
%
%    OP = PROJ_0( c , A) returns an implementation of the
%    indicator function of the set {x : Ax = c}
%    where A is a matrix. Unlike the previous call, c must now
%    be a vector of appropriate size, and x must be a matrix
%    (no support for multi-dimensional arrays).
%    Note:
%       This requires computing the Cholesky decomposition of A
%       which can be expensive. For large dimensions, consider
%       using TFOCS_SCD.m and explicitly passing in the linear
%       operator, which will recast the optimization problem to avoid
%       requiring this function. For small problems, directly
%       calling this function is fine and may lead to faster convergence.
%
%       If A is diagonal, then we can exploit that, but we don't 
%       currently have it implemented. Instead, just rescale c
%       since if A=di\ag(d) then { x : Ax=c } = { x : x = c./d }
%
%       "A" must be an explicit matrix, not a function handle
%
%   OP = PROJ_0( c , A, algo)
%       lets the user select between two algorithms (algo=1 or =2)
%       Algo. 1 is a bit faster but more prone to numerical roundoff error
%       which could be significant if A is extremely ill-conditioned
%
%   Note: for the set { x : ||Ax-b|| <= eps } where eps > 0, see
%       proj_l2.m
%
% See also proj_l2.m, prox_0.m, proj_Rn.m and smooth_constant.m,
%   which are the Fenchel conjugates of this function.

if nargin == 0
    op = @proj_0_impl;
elseif nargin == 1
    op = @(varargin) proj_0_impl_q( offset, varargin{:} );
elseif nargin >= 2 % added June 14 2014
    if norm(offset)==0
        error('offset term is 0, so solution is always 0. Are you sure this is what you want?');
    end
    if size( A, 1 ) ~= size(offset,1)
        error('A must have as many rows as c');
    end
    
    % We have two different algorithms, same asymptoptic complexity
    % but Algo. 1 is faster in practice though less numerically robust
    %   since it forms the matrix A*A' which has larger condition number
    % Algo. 2 is a bit more stable but a bit slower
    if nargin < 3 || isempty(algo), algo = 1; end
    if algo == 1
        disp('Now computing Cholesky factorization of A. Please wait...');
        R = chol( A*A' );
    elseif algo==2
        disp('Now computing QR decomposition of A^T. Please wait...');
        [Q,R] = qr(A',0);
    else
        error('Bad value for "algo". Should be 1 or 2');
    end
    disp('... done');
    op = @(varargin) proj_0_impl_qA( offset, A, R, algo, varargin{:} );
end

function [ v, x ] = proj_0_impl( x, t )
v = 0;
switch nargin,
	case 1,
		if nargout == 2,
			error( 'This function is not differentiable.' );
        elseif any( x(:) ),
            v = Inf;
        end
	case 2,
        % "t" variable has no effect
		x = 0*x;
	otherwise,
		error( 'Not enough arguments.' );
end
function [ v, x ] = proj_0_impl_q( c,  x, t )
v = 0;
switch nargin,
	case 2,
		if nargout == 2,
			error( 'This function is not differentiable.' );
        end
        if isscalar(c) 
            if any( x(:) - c )
                v = Inf;
            end
        elseif any( x(:)  - c(:) ),
            v = Inf;
        end
	case 3,
        % "t" variable has no effect
        if isscalar(c) && ~isscalar(x)
            x = c*ones( size(x) );
        else
            x = c;
        end

	otherwise,
		error( 'Not enough arguments.' );
end

function [ v, x ] = proj_0_impl_qA( b, A, R, algo, x, t )
v = 0;
switch nargin,
    case 5,
        if nargout == 2,
            error( 'This function is not differentiable.' );
        end
        if norm( A*x - b )/norm(b) > 1e-10
            v = Inf;
        end
    case 6,
        % "t" variable has no effect
        if algo==1
            AAtinv = @(x) R\( R'\x );
            x = x - A'*AAtinv( A*x - b );
        elseif algo==2
            bb = (R')\b;
            AA = @(x) (R')\(A*x);  % this is now Q'
            AAt= @(x) A'*(R\x);
            x  = x - AAt( AA(x) - bb );
        end
    otherwise,
        error( 'Not enough arguments.' );
end
% TFOCS v1.3 by Stephen Becker, Emmanuel Candes, and Michael Grant.
% Copyright 2013 California Institute of Technology and CVX Research.
% See the file LICENSE for full license information.
