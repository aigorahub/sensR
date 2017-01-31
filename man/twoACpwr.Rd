\name{twoACpwr}
\alias{twoACpwr}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Exact Power Computation for the 2-AC Discrimination Protocol}
\description{
  
  Computes the exact power for the 2-AC protocol using the (signed)
  likelihood root statistic. Power is computed for a significance test
  of d-prime. The \code{tol} argument specifies the
  precision with which power should be computed.
  
}
\usage{

twoACpwr(tau, d.prime, size, d.prime0 = 0, alpha = 0.05, tol = 1e-5, 
         return.dist = FALSE, statistic = "likelihood",
         alternative = c("two.sided", "less", "greater"))

}
\arguments{
  \item{tau}{the value of tau under the alternative hypothesis}
  \item{d.prime}{the value of d.prime under the alternative hypothesis}
  \item{size}{the sample size}
  \item{d.prime0}{the value of d-prime under the null hypothesis in the
    significance test for which power should be computed}
  \item{alpha}{the size of the test}
  \item{tol}{specifies the precision with which power should be
    computed, e.g., \code{1e-4} cause power to be computed correctly to
    three significant digits. Lower values of \code{tau} gives higher
    precision, but also longer computation times.}
  \item{return.dist}{should the p-value distribution be returned rather
    than the power be computed?}
  \item{statistic}{the statistic used in the significance test for which
    the power should be computed. Currently only the (signed) likelihood
    root statistic is available---se the details for more information.}
  \item{alternative}{the type of alternative hypothesis in the
    significance test for which the power should be computed}
}
\details{

  The main idea in this function is to compute all possible data
  outcomes and then compute the p-value for the chosen significance test
  for each of these outcomes. This gives the exact distribution of
  p-values from which the exact power can be computed. This is basically
  what happens if \code{tol = 0}.

  There is, however, a problem with this approach if \code{size} is
  large, since the the number of possible outcomes increases very fast
  with the \code{size}; the order is O(\code{n^2}). The solution to this
  problem is to ignore those outcomes which will occur with very small
  probability. Often, a large proportion of the outcomes, say 90\% will
  occur so rarely that they account for, say \code{1e-4} percent of the
  probability mass; it is therefore safe to ignore those outcomes
  without compromising the accuracy of the computed power by any
  relevant amount. For more information see the referenced paper and the
  package vignette Statistical Methodology. 
  

  The Wald statistic is not available here. The reason is that the Wald
  statistic is not always defined and the problem is therefore what to
  do with those cases where it is not defined?
  On the other hand the likelihood root statistic
  is defined in all cases, so theres is no problem here, and since the
  likelihood root statistic is more accurate than the Wald statistic,
  there is not much reason to use the Wald statistic after all.

  For the record; the Wald statistic is not defined, when the standard
  error of d-prime is not defined. This happens when the
  variance-covariance matrix of tau and d-prime is not defined, which
  occurs in a number of boundary cases, i.e., when one or more cells
  contain zero frequencies. Since these outcomes occur with positive
  probability, the algorithm used by \code{twoACpwr} will always
  encounter those cases and have to deal with them. This would be
  cumbersome to implement.
}
\seealso{
  \code{\link{clm2twoAC}}, \code{\link{twoACpwr}}
}
\value{
  A \code{data.frame} with one line and the following entries
  \item{power}{the computed power}
  \item{actual.alpha}{the actual size of the test (different from the
    nominal alpha given as argument due to the discreteness of the
    observations). }
  \item{samples}{the number of possible outcomes for this \code{size}}
  \item{discarded}{the number of outcomes for which the p-value is not
    computed. This number is zero if \code{tol = 0}}
  \item{kept}{the number of outcomes for which the p-value is computed
    in. This number equals \code{samples} if \code{tol = 0}}
  \item{p}{the probability vector of the multinomial distribution
    implied by the values of \code{tau} and \code{d.prime}.}
}
\references{
  Christensen R.H.B., Lee H-S and Brockhoff P.B. (2011). Estimation of
  the Thurstonian model for the 2-AC protocol. Submitted to Food
  Quality and Preference. 
}

\author{Rune Haubo B Christensen}

\examples{

## Exact power: 
twoACpwr(tau = .5, d.prime = .7, size = 50, tol = 0)

## Power exact to a reasonable number of digits
twoACpwr(tau = .5, d.prime = .7, size = 50, tol = 1e-5)

## Power for a similarity test in a discrimination setting where the
## true parameter values are expected to be tau = 0.4 and true d.prime
## = .5, while we want to show that d.prime < 1, i.e., under the null
## hypothesis d.prime = 1:
twoACpwr(tau = .4, d.prime = .5, size = 100, d.prime0 = 1, tol = 1e-5, 
         alternative = "less")

## Power for a difference test in a preference setting where the true
## parameter values are expected to be tau = 0.4 and d.prime = -0.5,
## while we want to show that d.prime is different from zero:
twoACpwr(tau = 0.4, d.prime = -0.5, size = 100, d.prime0 = 0, tol = 1e-5, 
         alternative = "two.sided")

}
\keyword{models}% at least one, from doc/KEYWORDS
