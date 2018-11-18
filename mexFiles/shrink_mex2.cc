/**
 * Implements an equivalent of
 * 		shrink = @(X, lambda) sign(X) .* max( abs(X) - lambda, 0 );
 *
 * 	but this should be faster and more memory efficient.
 *
 * 	This also implements the variant
 * 		shrink = @(X, lambda, offset) sign(X-offset) .* max( abs(X-offset) - lambda, 0 );
 *
 *	Inputs:
 *		Mode 1 (no offset):
 *			Y = shrink_mex2(X, lambda);
 *
 *		Mode 2 (with offset):
 *			Y = shrink_mex2(X, lambda, offset);
 *
 *		Mode 3 (set internal options):
 *			opt = struct('num_threads', 2);
 *			shrink_mex2(opt);
 *
 * Created on 17 Nov 2018 by jamesfolberth@gmail.com
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <sstream>

#if defined(__AVX__)
	#include <immintrin.h>
	#define HAVE_AVX 1
#else
	#define HAVE_AVX 0
#endif

#include <omp.h>

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

void shrink(double *y,
			const double *x,
			const double lambda,
			size_t n) {

	#if HAVE_AVX
	if (n >= 4) {
		__m256d vlambda = _mm256_set1_pd(lambda);
		__m256d sign_bit_mask = _mm256_set1_pd(-0.);
		for ( ; n >= 4; n -= 4) {
			__m256d vx = _mm256_loadu_pd(x);

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

	for ( ; n != 0 ; --n) {
		*y = std::copysign(std::max(std::abs(*x) - lambda, 0.), *x);
		++x;
		++y;
	}
}

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

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	// Check inputs
	if (nrhs < 1) {
		mexErrMsgIdAndTxt(
				"TFOCS:shrink_mex2:tooFewInputs",
				"At least one input is required.");
	}

	if (nrhs > 3) {
		mexErrMsgIdAndTxt(
				"TFOCS:shrink_mex2:tooManyInputs",
				"No more than three inputs.");
	}

	if (nlhs > 1) {
		mexErrMsgIdAndTxt(
				"TFOCS:shrink_mex2:tooManyOutputs",
				"Exactly one output.");
	}

	// Cached options struct
	static Options opt;

	// First input
	if (mxIsStruct(prhs[0])) {
		opt.parseStruct(prhs[0]);
		return;
	}

	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || mxIsSparse(prhs[0])) {
		mexErrMsgIdAndTxt(
				"TFOCS:shrink_mex2:invalidX",
				"Input matrix should be real, full (non-sparse), double X.");
	}

	size_t m = mxGetM(prhs[0]);
	size_t n = mxGetN(prhs[0]);
	const double *X = mxGetPr(prhs[0]);

	// Second input
	//TODO JMF 18 Nov 2018: should accept a vector input?
	const double lambda = mxGetScalar(prhs[1]);

	// Allocate output
	plhs[0] = mxCreateUninitNumericMatrix(m, n, mxDOUBLE_CLASS, mxREAL);
	double *Y = mxGetPr(plhs[0]);

	// Parse the third input
	// Could be empty, or offset (scalar/vector)
	const mxArray *offset_array = nullptr;
	if (nrhs == 3) {
		offset_array = prhs[2];
	}

	// Okay, now we're ready to go
	//std::cout << "opt.num_threads = " << opt.num_threads << std::endl;
	if (!offset_array) {
		#pragma omp parallel for num_threads(opt.num_threads) schedule(static)
		for (size_t j=0; j<n; ++j) {
			shrink(Y + m*j, X + m*j, lambda, m);
		}

	} else {
		size_t n_offset = mxGetNumberOfElements(offset_array);
		if (n_offset > 1) {
			if (n_offset != n) {
				mexErrMsgTxt("Offset should be a scalar or vector of size "
						"equal to number of columns of X.");
			}

			double *offset = mxGetPr(offset_array);

			#pragma omp parallel for num_threads(opt.num_threads) schedule(static)
			for (size_t j=0; j<n; ++j) {
				shrink_offset(Y + m*j, X + m*j, lambda, offset[j], m);
			}

		} else {
			double offset = mxGetScalar(offset_array);
			#pragma omp parallel for num_threads(opt.num_threads) schedule(static)
			for (size_t j=0; j<n; ++j) {
				shrink_offset(Y + m*j, X + m*j, lambda, offset, m);
			}
		}
	}
}

