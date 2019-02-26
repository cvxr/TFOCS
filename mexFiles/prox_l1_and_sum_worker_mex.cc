/**
 *
 * Implements a worker routine for prox_l1_and_sum*.m
 *
 * Inputs:
 * 		Mode 1 (vector):
 * 			% X is a vector
 * 			Y = prox_l1_and_sum_worker_mex(X, lambda, b)
 *
 * 		Mode 2 (matrix):
 * 			% X is [nRows x nCols]
 * 			Y = prox_l1_and_sum_worker_mex(X, lambda, b)
 *
 * 		Mode 3 (matrix, zero diagonal):
 * 			% X is [nRows x nCols]
 * 			% zeroID = true enforces diag(Y) == 0
 * 			Y = prox_l1_and_sum_worker_mex(X, lambda, b, zeroID)
 *
 * 		Mode 4 (set internal options):
 * 			opt = struct('num_threads', 2);
 * 			prox_l1_and_sum_worker_mex(opt);
 *
 * Created on 18 Nov 2018 by jamesfolberth@gmail.com
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <limits>

#if defined(__AVX__)
    #include <immintrin.h>
    #define HAVE_AVX 1
#else
    #define HAVE_AVX 0
#endif
#if defined(_OPENMP)
#include <omp.h>
#endif

#include <mex.h>

struct Options {
    Options() : num_threads(1) {}

    int num_threads;

    void parseStruct(const mxArray *marr);
};

void Options::parseStruct(const mxArray *mstruct) {
    // Check field names
    int n_fields = mxGetNumberOfFields(mstruct);
    for (int i=0; i<n_fields; ++i) {
        std::string name = mxGetFieldNameByNumber(mstruct, i);
        if (name.compare("num_threads") == 0) continue;
        else {
            std::string msg = "Unrecognized options field name: " + name;
            mexErrMsgTxt(msg.c_str());
        }
    }

    mxArray* tmp;
    tmp = mxGetField(mstruct, 0, "num_threads");
    if (nullptr != tmp) {
        double dval = mxGetScalar(tmp);
        if (dval == std::round(dval)) {
            num_threads = static_cast<int>(std::round(dval));
        } else {
            mexErrMsgTxt("opt.num_threads should be an integer.");
        }
}
}

// Copied from shrink_mex2.cc
void shrink_offset(double *y,
                   const double *x,
                   const double lambda,
                   const double offset,
                   size_t n) {

    #if HAVE_AVX
    if (n >= 4) {
        __m256d vlambda = _mm256_set1_pd(lambda);
        __m256d voffset = _mm256_set1_pd(offset);
        __m256d sign_bit_mask = _mm256_set1_pd(-0.);
        for ( ; n >= 4; n -= 4) {
            __m256d vx = _mm256_loadu_pd(x);
            vx = _mm256_sub_pd(vx, voffset);

            __m256d xabs = _mm256_andnot_pd(sign_bit_mask, vx);
            __m256d max = _mm256_max_pd(_mm256_sub_pd(xabs, vlambda), _mm256_setzero_pd());
            __m256d xsign_bits = _mm256_and_pd(vx, sign_bit_mask);
            __m256d vy = _mm256_or_pd(xsign_bits, max);

            _mm256_storeu_pd(y, vy);

            x += 4;
            y += 4;
        }
    }
    #endif

    for ( ; n != 0; --n) {
        *y = std::copysign(std::max(std::abs(*x - offset) - lambda, 0.), *x - offset);
        ++x;
        ++y;
    }
}

class Proxl1Sum {
public:
    // To do the vector and matrix case w/o zero_diag, set zero_ind < 0
    // To enforce y[zero_ind] == 0, set zero_ind >= 0; this is used in the
    // matrix case when zero_diag is true.
    void run(double *y,
             const double *x0,
             size_t n,
             double lambda,
             double b,
             ptrdiff_t zero_ind=-1);

private:
    // Evaluates the y = shrink(x0 - a, lambda)
    void evaluate_prox(double *y,
                       const double *x0,
                       size_t n,
                       double lambda,
                       double a);

    // Evaluates sum(y) - b
    double evaluate_func(double *y,
                         size_t n,
                         double b);

    std::vector<double> break_points_;
};

void Proxl1Sum::run(double *y,
                    const double *x0,
                    size_t n,
                    const double lambda,
                    const double b,
                    ptrdiff_t zero_ind) {
    // Allocate memory
    // On subsequent calls, these will stay allocated
    // On destruction (e.g. after this mex routine returns), they'll get freed
    break_points_.resize(2*n);

    // Set break_points = [x0 - lambda; x0 + lambda]
    for (size_t i=0; i<n; ++i) {
        break_points_[i] = x0[i] - lambda;
        break_points_[i+n] = x0[i] + lambda;
    }

    // We ignore the diagonal inds by setting the diagonal break points to -inf
    // so the sort pushes them to the bottom of break_points_.  We then do the
    // bisection search taking this into account.
    if (zero_ind >= 0) {
        break_points_[zero_ind] = -std::numeric_limits<double>::infinity();
        break_points_[zero_ind+n] = -std::numeric_limits<double>::infinity();
    }

    // Sort break_points in decreasing order
    std::sort(break_points_.rbegin(), break_points_.rend());

    // Bisection search to find sum(x) - b == 0
    ptrdiff_t lower_bound = -1;
    size_t num_break_points = (zero_ind >= 0) ? 2*n - 2 : 2*n;
    ptrdiff_t upper_bound = num_break_points;
    size_t n_its = 1 + std::ceil(std::log2(static_cast<double>(num_break_points)));

    ptrdiff_t ind = lower_bound;
    for (size_t it=0; it<n_its; ++it) {
        // Check if we've closed in on the break point
        if (upper_bound - lower_bound <= 1) {
            break;
        }

        // Bisect
        ind = std::round(0.5*(static_cast<double>(lower_bound)
                              + static_cast<double>(upper_bound)));

        // Check that we're not at either endpoint (which may be invalid indexes)
        if (ind == lower_bound) {
            ++ind;
        } else if (ind == upper_bound) {
            --ind;
        }

        // Evaluate the prox at this offset and check the function value
        double a = break_points_[ind];
        evaluate_prox(y, x0, n, lambda, a);
        double h = evaluate_func(y, n, b);

        if (zero_ind >= 0) {
            h -= y[zero_ind];
        }

        // Pick one of the sections
        if (h > 0) {
            upper_bound = ind;
        } else {
            lower_bound = ind;
        }
    }

    // Now determine linear part, which we infer from two points.
    // If the lower or upper bounds are infinite, we take special care
    // by using a new "a" that is slightly bigger/lower, respectively.
    // This is then used to extract the linear part.
    double a;
    if (lower_bound == -1) {
        a = break_points_[upper_bound];
        // use a - 10 as lower bound
        a = 0.5*(a - 10 + a);
    } else if (upper_bound == num_break_points) {
        a = break_points_[lower_bound];
        // use a + 10 as upper bound
        a = 0.5*(a + a + 10);
    } else {
        // general case
        a = 0.5*(break_points_[lower_bound] + break_points_[upper_bound]);
    }

    // Now we have the support; find the exact value
    evaluate_prox(y, x0, n, lambda, a); // to find the support

    double nu = 0.;
    size_t n_supp = 0;
    for (size_t i=0; i<n; ++i) {
        // only want the support
        // use abs in case shrink_offset creates -0's
        if (std::abs(y[i]) == 0.) {
            continue;
        }

        nu += x0[i] - (std::signbit(y[i]) ? -lambda : lambda);
        ++n_supp;
    }

    // Don't count the diagonal
    if (zero_ind >= 0) {
        if (std::abs(y[zero_ind]) != 0.) {
            nu -= x0[zero_ind] - (std::signbit(y[zero_ind]) ? -lambda : lambda);
            --n_supp;
        }
    }

    nu -= b;
    nu /= n_supp;

    evaluate_prox(y, x0, n, lambda, nu);

    // Set the diagonal to 0
    if (zero_ind >= 0) {
        y[zero_ind] = 0.;
    }

}

void Proxl1Sum::evaluate_prox(double *y,
                              const double *x0,
                              size_t n,
                              double lambda,
                              double a) {
    shrink_offset(y, x0, lambda, a, n);
}

double Proxl1Sum::evaluate_func(double *y,
                                size_t n,
                                double b) {
    double h = -b;
    for (size_t i=0; i<n; ++i) {
        h += y[i];
    }

    return h;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Check inputs
    if (nrhs < 1) {
        mexErrMsgTxt("At least one input is required");
    }

    if (nrhs > 4) {
        mexErrMsgTxt("No more than four inputs");
    }

    if (nlhs > 1) {
        mexErrMsgTxt("Exactly one output");
    }

    // Cached options struct
    static Options opt;

    // First input
    if (mxIsStruct(prhs[0])) {
        opt.parseStruct(prhs[0]);
        return;
    }

    if (nrhs < 3) {
        mexErrMsgTxt("Must specify inputs X, lambda, b");
    }

    if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxIsSparse(prhs[0])) {
        mexErrMsgTxt("X should be real, full (non-sparse), double");
    }

    size_t nRows = mxGetM(prhs[0]);
    size_t nCols = mxGetN(prhs[0]);
    const double *x = mxGetPr(prhs[0]);

    // Second input
    // lambda can be a vector or a scalar
    size_t n_lambda = mxGetNumberOfElements(prhs[1]);
    if (n_lambda > 1 && n_lambda != nCols) {
        mexErrMsgTxt("lambda should be a scalar or have number of elements "
                "equal to the number of columns of X.");
    }
    const double *lambda = mxGetPr(prhs[1]);



    // Third input
    const double b = mxGetScalar(prhs[2]);

    // Fourth input (optional)
    bool zero_diag = false;
    if (nrhs == 4) {
        if (!mxIsLogicalScalar(prhs[3])) {
            mexErrMsgTxt("zeroID should be true or false");
        }
        zero_diag = mxIsLogicalScalarTrue(prhs[3]);
    }

    // Allocate output
    plhs[0] = mxCreateUninitNumericMatrix(nRows, nCols, mxDOUBLE_CLASS, mxREAL);
    double *y = mxGetPr(plhs[0]);

    // Okay, now we're ready to go
    if (nCols == 1) { // vector case
        Proxl1Sum prox;
        prox.run(y, x, nRows, lambda[0], b);

    } else {
        #pragma omp parallel num_threads(opt.num_threads)
        {
            Proxl1Sum prox;

            #pragma omp for schedule(static)
            for (size_t j=0; j<nCols; ++j) {
                double lambda_val = (n_lambda > 1) ? lambda[j] : lambda[0];
                if (zero_diag) {
                    prox.run(y + nRows*j, x + nRows*j, nRows, lambda_val, b, j);
                } else {
                    prox.run(y + nRows*j, x + nRows*j, nRows, lambda_val, b);
                }
            }
        }
    }
}
