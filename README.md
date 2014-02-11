# TFOCS: Templates for First-Order Conic Solvers

TFOCS (pronounced *tee-fox*) provides a set of Matlab templates, or building
blocks, that can be used to construct efficient, customized sovlers for a
variety of convex models, including in particular those employed in sparse
recovery applications. It was conceived and written by [Stephen
Becker](http://ugcs.caltech.edu/~srbecker/),  [Emmanuel J.
Candès](http://statweb.stanford.edu/~candes/) and  [Michael
Grant](http://cvxr.com/bio).

More information about the software can be found in the
[paper](https://github.com/cvxr/TFOCS/raw/master/TFOCS.pdf) and the 
[users' guide](https://github.com/cvxr/TFOCS/blob/master/userguide.pdf).

TFOCS is jointly owned by [CVX Research, Inc.](http://cvxr.com) and
[Caltech](http://caltech.edu). As of October 2, 2013, it has been made freely
available for both academic and commercial use under a 
[BSD 3-Clause license](https://github.com/cvxr/TFOCS/raw/master/LICENSE),
subject to requirements of attribution and non-endorsement, 
and a disclaimer of warranty contained therein.

## Downloading

The source code for TFOCS is now hosted on
[GitHub](https://github.com/mcg1969/TFOCS). Feel free to visit the GitHub page
to browse the code or clone the repository. Direct download links for the
latest versions are provided on the
[Releases](https://github.com/mcg1969/TFOCS/releases) page. The packages
include the main program files, the documentation, the paper, and a number of
examples and demos.

## Documentation / Demos
 
The users' guide is included with the distribution, and available separately
[here](https://github.com/cvxr/TFOCS/raw/master/userguide.pdf).  
[Click here](http://cvxr.com/tfocs/functions/) for a list of the functions
included with TFOCS.  
Several demonstrations are available [here](http://cvxr.com/tfocs/demos/).
Each demonstration includes the source code and data files needed to reproduce
the results.

## Paper

[Templates for convex cone problems ith applications to sparse signal
recovery](https://github.com/cvxr/TFOCS/raw/gh-pages/TFOCS.pdf)  
by [S. Becker](http://ugcs.caltech.edu/~srbecker/), [E.
Candès](http://statweb.stanford.edu/~candes/), and [M.
Grant](http://cvxr.com/bio).  
Stanford University Technical Report, September 2010.  
Published in 
[Mathematical Programming Computation](http://mpc.zib.de/index.php/MPC/article/view/58), Volume 3,
Number 3, August 2011.

In the spirit of [reproducible research](http://reproducibleresearch.net/), we
provide the 
[source code and data](https://github.com/cvxr/TFOCS/raw/gh-pages/TFOCS_paperExamplesOnly.zip) 
(ZIP file, 9.7MB) used for the examples in the paper. You
must have the TFOCS package installed to run these examples.

### Abstract

This paper develops a general framework for solving a variety of convex cone
problems that frequently arise in signal processing, machine learning,
statistics, and other fields. The approach works as follows: first, determine
a conic formulation of the problem; second, determine its dual; third, apply
smoothing; and fourth, solve using an optimal first-order method. A merit of
this approach is its flexibility: for example, all compressed sensing problems
can be solved via this approach. These include models with objective
functionals such as the total-variation norm, ||Wx||_1 where W is arbitrary, or a
combination thereof. In addition, the paper also introduces a number of
technical contributions such as a novel continuation scheme, a novel approach
for controlling the step size, and some new results showing that the smooth
and unsmoothed problems are sometimes formally equivalent. Combined with our
framework, these lead to novel, stable and computationally efficient
algorithms. For instance, our general implementation is competitive with
state-of-the-art methods for solving intensively studied problems such as the
LASSO. Further, numerical experiments show that one can solve the Dantzig
selector problem, for which no efficient large-scale solvers exist, in a few
hundred iterations. Finally, the paper is accompanied with a software release.
This software is not a single, monolithic solver; rather, it is a suite of
programs and routines designed to serve as building blocks for constructing
complete algorithms.

## Support

Please note that the authors of TFOCS generally cannot offer direct email
assistance. We hope that the resources we have assembled here will address
most users’ issues. If the documentation and demos are insufficient, please
consider the following alternatives:

### Community-driven support

For *TFOCS-specific* usage questions that are not caused by bugs, please
consider posing your question on the [CVX Forum](http://ask.cvxr.com), a free
question and answer forum modeled in the style of StackExchange family of
sites. As the name implies, the forum is also used for
[CVX](http://cvxr.com/cvx), but TFOCS users are welcome to ask questions on
this forum as well. The authors of TFOCS do visit this forum regularly. If you
are an experienced TFOCS user, we would ask you to join us and offer answers
to other TFOCS users’ questions.

For more general questions about optimization that are not specific to TFOCS,
the CVX Forum is not the appropriate venue. Instead, we invite you to consider
two other online forums: the [Computational Science Stack Exchange (CompSci
SE)](http://scicomp.stackexchange.com/) and [OR Exchange](http://or-
exchange.org/). CompSci SE covers a variety of computational science topics,
including convex optimization. OR-Exchange is a Q&A site sponsored by
[INFORMS](https://www.informs.org/), an international society for
professionals in operations research, management science, and analytics. Both
communities include a number of active optimization experts.

### Submitting bug reports

If you encounter a bug in TFOCS, or an error in the documentation, please
submit an report on the [GitHub issue tracker for
TFOCS](https://github.com/cvxr/TFOCS/issues). In order for us to
effectively evaluate a bug report, we will need the following information:

* The output of the tfocs_version command, which provides information about
  your operating system, your MATLAB version, and your TFOCS version. Just copy
  and paste this information from your MATLAB command window into your report. 
* A description of the error itself. If TFOCS provided the error message,
  please copy the full text of the error output into the report. 
* If it is at all possible, please provide us with a brief code sample and
  supporting data that reproduces the error. If that cannot be accomplished,
  please provide a detailed description of the circumstances under which the error occurred.

### Mailing list

If you wish to be notified of new TFOCS releases, please join the [google
group mailing list](https://groups.google.com/forum/#!forum/tfocs). Note that
this mailing list is for announcements only.

### Contract support

For more in-depth support, you may purchase a support contract from one or
more of the authors. Please contact tfocs@cvxr.com for more information.



