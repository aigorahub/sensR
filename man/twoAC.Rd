\name{twoAC}
\alias{twoAC}
\alias{print.twoAC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{2-AC Discrimination and Preference Protocol}
\description{
  Computes estimates and standard errors of d-prime and tau for the two
  alternative (2-AC) protocol. A confidence interval and
  significance test for d-prime is also provided. The 2-AC protocol is 
  equivalent to a 2-AFC protocol with a "no-difference" option, and
  equivalent to a paired preference test with an "no-preference" option. 
}
\usage{

twoAC(data, d.prime0 = 0, conf.level = 0.95,
      statistic = c("likelihood", "Wald"),
      alternative = c("two.sided", "less", "greater"), ...)

}
\arguments{
  \item{data}{
    a non-negative numeric vector of length 3 with the number of
    observations in the three response categories in the form ("prefer
    A", "no-preference", "prefer B"). If the third element
    is larger than the first element, the estimate of d-prime is
    positive.} 
  \item{d.prime0}{the value of d-prime under the null hypothesis for the
    significance test.}
  \item{conf.level}{the confidence level.}
  \item{statistic}{the statistic to use for confidence level and
    significance test.}
  \item{alternative}{the type of alternative hypothesis.}
  \item{\dots}{not currently used.}
}
\details{
  \code{\link[=confint.twoAC]{confint}},
  \code{\link[=profile.twoAC]{profile}},
  \code{logLik}, \code{vcov}, and
  \code{print} methods are implemented for \code{twoAC} objects.

  Power computations for the 2-AC protocol is implemented in
  \code{\link{twoACpwr}}. 
}
\seealso{
  \code{\link{clm2twoAC}}, \code{\link{twoACpwr}}
}
\value{
  An object of class \code{twoAC} with elements
  \item{coefficients}{
    2 by 2 coefficient matrix with estimates and standard errors of
    d-prime and tau. If the variance-covariance matrix of the parameters
    is not defined, the standard errors are \code{NA}.
  }
  \item{vcov}{
    variance-covariance matrix of the parameter estimates. Only present
    if defined for the supplied data.
  }
  \item{data}{
    the data supplied to the function.
  }
  \item{call}{
    the matched call.
  }
  \item{logLik}{
    the value of the log-likelihood at the maximum likelihood estimates.
  }
  \item{alternative}{the name of the alternative hypothesis for the
    significance test.}
  \item{statistic}{the name of the test statistic used for the
    significance test.}
  \item{conf.level}{the confidence level for the confidence interval for
    d-prime.}
  \item{d.prime0}{the value of d-prime under the null hypothesis in the
    significance test.}
  \item{p.value}{p-value of the significance test.}
  \item{confint}{two-sided condfidence interval for d-prime. This is
    only available if the standard errors are defined, which may happen
    in boundary cases. Use \code{profile} and \code{confint} methods to
    get confidence intervals instead; see the examples.}
}
\references{
  Christensen R.H.B., Lee H-S and Brockhoff P.B. (2011). Estimation of
  the Thurstonian model for the 2-AC protocol. Submitted to Food
  Quality and Preference. 
}

\author{Rune Haubo B Christensen}

\examples{

## Simple:
fit <- twoAC(c(2,2,6))
fit

## Typical discrimination-difference test: 
(fit <- twoAC(data = c(2, 5, 8), d.prime0 = 0, alternative = "greater"))

## Typical discrimination-similarity test: 
(fit <- twoAC(data = c(15, 15, 20), d.prime0 = .5, alternative = "less"))

## Typical preference-difference test:
(fit <- twoAC(data = c(3, 5, 12), d.prime0 = 0,
              alternative = "two.sided"))

## Typical preference (non-)inferiority test:
(fit <- twoAC(data = c(3, 5, 12), d.prime0 = 0,
              alternative = "greater"))

## For preference equivalence tests (two-sided) use CI with alpha/2:
## declare equivalence at the 5% level if 90% CI does not contain,
## e.g, -1 or 1: 
(fit <- twoAC(data = c(15, 10, 10), d.prime0 = 0, conf.level = .90))

## The var-cov matrix and standard errors of the parameters are not
## defined in all situations. If standard errors are not
## defined, then confidence intervals are not provided directly:
(fit <- twoAC(c(5, 0, 15)))
## We may use profile and confint methods to get confidence intervals
## never the less: 
pr <- profile(fit, range = c(-1, 3))
confint(pr)
plot(pr)

}
\keyword{models}% at least one, from doc/KEYWORDS
