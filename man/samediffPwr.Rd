\name{samediffPwr}
\alias{samediffPwr}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Power Analysis for Same-different Experiments}
\description{
  Computes the power for at same-different discrimination experiment
  with a no-difference null hypothesis via simulation.
}
\usage{
samediffPwr(n = 1000, tau, delta, Ns, Nd, alpha = 0.05)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{n}{the number of samples to use in the simulation. More samples
    means higher precision, but takes longer to compute.}
  \item{tau}{the value of tau}
  \item{delta}{the underlying sensory difference under the \emph{alternative}
    hypothesis (non-negative)}
  \item{Ns}{the number of same-samples (a positive integer)}
  \item{Nd}{the number of different-samples (a positive integer)}
  \item{alpha}{the type I level of the test (must be between zero and
    one)}
}
\details{
  The power is computed using simulations. \code{n} datasets is
  simulated from the
  Same Different model with specified parameters. The power is the
  fraction of times the p-value is lower than \code{alpha}.

  Under some parameter combinations, there is a non-significant
  probability that data will fall, so that the MLE of \code{delta} is
  not defined and the p-value is not defined. All such undefined
  p-values are silently ignored.

  The estimated power may change between runs and especially if the
  power is either very large or very small (ie. close to 0 or 1). Using
  more simulations will provide higher accuracy.

  It is often a good idea to run the power simulation a couple of times
  to ensure that the variation in the result is acceptable.
}
\value{
  A single numeric value giving the power of the specified test.
}
\references{Christensen, R.H.B., Brockhoff, P.B. (2009).
  Estimation and inference in the same-different test.
  Food, Quality and Preference, 20 pp. 514--520}
\author{Rune Haubo B Christensen}

\seealso{
  \code{\link{samediff}}, \code{\link{samediffSim}}
}

\examples{
## Finding the power of a discrimination test with a sensory delta of 2
## (alternative hypothesis) versus a null hypothesis of delta = 0 with
## a sample of size 2 x 10 and a type I level of .05. n should be higher
## for a reasonable precision:

samediffPwr(n = 100, tau = 1, delta = 2, Ns = 10, Nd = 10)

}
\keyword{models}% at least one, from doc/KEYWORDS
