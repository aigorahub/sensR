\name{dod_utils}
\alias{optimal_tau}
\alias{par2prob_dod}
\alias{dod_nll}
\alias{dod_null}
\alias{dod_null_tau}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Utility functions for the Degree-of-Difference model}
\description{
  Various utility functions supporting the Degree-of-Difference (DOD)
  model.
}
\usage{

optimal_tau(d.prime, d.prime0 = 0, ncat=3,
            method=c("equi.prob", "LR.max", "se.min"),
            tau.start=NULL, equi.tol = 1e-4, grad.tol = 1e-2,
            do.warn=TRUE)

par2prob_dod(tau, d.prime)

dod_nll(tau, d.prime, same, diff, integer.tol=1e-8)

dod_null(same, diff, integer.tol=1e-8)

dod_null_tau(same, diff)

}
\arguments{
  \item{d.prime}{the value of d-prime; non-negative numerical scalar.}
  \item{d.prime0}{d-prime under the null hypothesis; only used in
    \code{optimal_tau} when \code{method = "LR.max"}.}
  \item{ncat}{the number of response categories in the DOD model.}
  \item{method}{the method with which to choose the boundary
    parameters --- see \code{\link{dodSim}} for details on the methods.}
  \item{tau.start}{optional vector of starting values.}
  \item{equi.tol}{convergence tolerence for the \code{"equi.prob"}
    method.}
  \item{grad.tol}{gradient convergence tolerence.}
  \item{do.warn}{issue warning if estimation of optimal tau does not
    converge?}
  \item{same}{The answers to same-pairs; either 1) a numeric vector of
    counts of length equal to the number of response categories
    ordered appropriately or 2) a
    factor where the levels indicate the response categories.}
  \item{diff}{the answers to different-pairs in the same format as
    \code{same}.}
  \item{tau}{vector of boundary parameters in the DOD model.}
  \item{integer.tol}{tolerence for when \code{same} or \code{diff}
    arguments are considered non-integer counts: a warning is issued if
    non-integer counts are encountered.}
}
%% \details{
%% }
\value{
  \item{\code{optimal_tau}}{computes optimal boundary parameters (tau)
    using various criteria.}
  \item{\code{par2prob_dod}}{computes the multinomial probability vectors from
    DOD model parameters.}
  \item{\code{dod_nll}}{implements the negative log-likelihood function for the
    DOD model.}
  \item{\code{dod_null}}{implements the negative log-likelihood function for
    the DOD model where d-prime = 0.}
  \item{\code{dod_null_tau}}{Estimates tau for the DOD model where
    d-prime = 0.}
}

%% \references{
%%   Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian
%%   models for sensory discrimination tests as generalized linear models.
%%   Food Quality and Preference, 21, pp. 330-338.
%% }
\author{Rune Haubo B Christensen}

\seealso{ \code{\link{dod}}, \code{\link{dod_fit}},
  \code{\link{dodSim}}, \code{\link{dodPwr}},
  \code{\link{dodControl}}
}

\examples{

## Compute optimal boundary parameters using the LR.max criterion for
## d.prime=1 with 4 categories:
dp <- 1
(Tau <- optimal_tau(d.prime=dp, d.prime0 = 0, ncat=4,
                    method="LR.max")$tau)
##  [1] 1.244642 2.109140 3.098985
## This set of boundary parameters optimize the power of the DOD test
## with d.prime = 1 under the alternative hypothesis.

## Compute the probability that an observation will fall in each of
## the (here 2*4=8) response categories given values of tau and d.prime:
par2prob_dod(tau=Tau, d.prime=dp)
##              [,1]      [,2]      [,3]       [,4]
##  p.same 0.6211921 0.2429480 0.1074307 0.02842911
##  p.diff 0.5124361 0.2571691 0.1596425 0.07075227

## Compute the negative log-likelihood given data and parameters:
Same <- c(10, 20, 30, 20)
Diff <- c(10, 10, 20, 40)
dod_nll(tau=Tau, d.prime=dp, same=Same,
        diff=Diff)
##  [1] 334.0986

## Compute the negative log-likelihood under the null hypothesis
## (where d.prime = 0):
dod_null(same=Same, diff=Diff)
##  [1] 208.8154
##  ## The boundary parameters for this:
(Tau0 <- dod_null_tau(same=Same, diff=Diff))
##  [1] 0.2224709 0.5688675 1.2546147

## Some equalities:
stopifnot(
    dod_nll(tau=Tau0, d.prime=0, same=Same, diff=Diff) ==
    dod_null(same=Same, diff=Diff))
stopifnot(
    dod_null(same=Same, diff=Diff) ==
    -dod_fit(same=Same, diff=Diff, d.prime=0)$logLik
    )
stopifnot(
    dod_nll(same=Same, diff=Diff, tau=Tau, d.prime=dp) ==
    -dod_fit(same=Same, diff=Diff, tau=Tau, d.prime=dp)$logLik
    )
stopifnot(all(
    dod_null_tau(same=Same, diff=Diff) ==
    dod_fit(Same, Diff, d.prime=0)$tau))

\dontshow{
## set.seed(1)
## a <- dodSim(sample.size = 10, replicates = 3, d.prime = 2,
##            method = "triangle", sd.indiv = 1)
## expect <- c(3L, 3L, 2L, 3L, 3L, 1L, 3L, 3L, 1L, 2L)
## stopifnot(all(a == expect))
## }
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{models}
