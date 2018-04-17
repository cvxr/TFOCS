/*
 * Implements an equivalent of:
 *  shrink = @(X,lambda) sign(X).*max( abs(X) - lambda, 0 );
 *
 * but much more memory efficient, and about as fast (and faster for large,
 * memory-limited systems)
 *
 * Variant:
 *  shrink = @(X,lambda,offset) sign(X-offset).*max( abs(X-offset) - lambda, 0 );
 * 
 *
 * eg. x = randn(8e3);
 *tic; y = shrink( x, .3 ); toc
        Elapsed time is 1.200497 seconds.
 *tic; yy = shrink_mex( x, .3 ); toc
        Elapsed time is 0.418361 seconds.
 *
 *
 * No special compilation instructions needed
 *  Just do:  mex shrink_mex.c (assuming you've setup your mex/compiler)
 *
 * Stephen Becker, 3/15/18
 * */


#include <math.h>
#include "mex.h"

/* Input Arguments */
#define	X_IN	prhs[0]
#define	LAMBDA	prhs[1]
#define	OFFSET	prhs[2]


/* Output Arguments */
#define	Y_OUT	plhs[0]


#if !defined(MAX)
#define MAX(A, B)   ((A) > (B) ? (A) : (B))
#endif

/*
 * http://hpac.rwth-aachen.de/teaching/pp-16/material/08.OpenMP-4.pdf
 * - Aliasing issues
 *  double * __restrict__ a
 * - Alignment
 *  __assume_aligned(a, 32); // Intel
 * -
 *-ffast-math /fp:fast -fp-model fast=2
 * - inline functions
 *#pragma omp declare simd uniform(a) linear(1: b)
 *
 *file:///Users/srbecker/Downloads/SIMD%20Vectorization%20with%20OpenMP.PDF
 *  AVX is 4 doubles (256 bit), AVX-512/MIC is twice that (SSE is half that)
 * "Before OpenMP 4.0..." 
 * */
void shrink( const size_t n, const double lambda, const double *x, double *y ) {
    size_t i;
    /* #pragma omp simd */
    /*#pragma omp parallel for simd*/
    /*#pragma omp simd aligned(a, b: 32)*/
    for ( i=0; i < n; i++ ){
        y[i] = copysign( MAX( 0.0, fabs(x[i]) - lambda ), x[i] );
        /* How much speedup from SIMD could we expect? 
          Not much. Even this simple code takes about the same time */
        /* y[i] = 3.0; */
    }
    
    /* copysign( double x, double y ); C99 standard
     * Composes a floating point value with the magnitude of x and the sign of y.
     * see also fabs, signbit
     * */
}

void shrink_offset( const size_t n, const double lambda, const double *x, double *y, const double offset ) {
    size_t i;
    for ( i=0; i < n; i++ )
        y[i] = copysign( MAX( 0.0, fabs(x[i]-offset) - lambda ), x[i]-offset );
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *xp;  /* pointer to input array x */
    double *yp;  /* pointer to output array y */
    double lambda, offset, *offset_ptr;
    size_t m,n, j;
    
    
    /* Check for proper number of arguments */
    
    if (nrhs < 2) { 
	    mexErrMsgIdAndTxt( "MATLAB:shrink_mex:invalidNumInputs",
                "Two to three input arguments required."); 
    } else if (nlhs > 1) {
	    mexErrMsgIdAndTxt( "MATLAB:shrink_mex:maxlhs",
                "Too many output arguments."); 
    } 
    
    if (!mxIsDouble(X_IN) || mxIsComplex(X_IN) || mxIsSparse(X_IN) ) { 
	    mexErrMsgIdAndTxt( "MATLAB:shrink_mex:invalidX",
                "SHRINK_MEX requires a real non-sparse double input."); 
    } 
    /* do these after verifying nrhs >= 2, otherwise error! */
    m = mxGetM(X_IN);
    n = mxGetN(X_IN); 
    lambda = mxGetScalar(LAMBDA);
    /* Create a matrix for the return argument */ 
    Y_OUT  = mxCreateDoubleMatrix( (mwSize)m, (mwSize)n, mxREAL); 
    
    
    /* Assign pointers to the various parameters */ 
    yp     = mxGetPr(Y_OUT);
    xp     = mxGetPr(X_IN);
    
    if ( nrhs > 2 ) {
        /* the offset could be a scalar or vector */
        if ( mxGetNumberOfElements(OFFSET) > 1 ) {
            if ( mxGetNumberOfElements(OFFSET) != n ) {
                mexErrMsgIdAndTxt( "MATLAB:shrink_mex:invalidOffset",
                        "Offset should be scalar or vector of size equal to number of columns of main input.");
            }
            offset_ptr = mxGetPr(OFFSET);
            for (j=0; j<n; j++)
                shrink_offset( m, lambda, xp+j*m, yp+j*m, offset_ptr[j] );
            
        } else {
            offset = mxGetScalar(OFFSET);
            shrink_offset( m*n, lambda, xp, yp, offset );
        }
    } else {
        shrink( m*n, lambda, xp, yp );
    }
        
    
    return;
    
}
