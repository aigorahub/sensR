\name{discrimSS}
\alias{discrimSS}
\alias{d.primeSS}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Sensory discrimination sample size calculation}
\description{
  Computes the sample size for a difference or similarity test for a
  sensory discrimination experiment using the binomial distribution.
  \code{d.primeSS} is a convenience function that calls
  \code{discrimSS} but has arguments in terms of d-prime rather than
  pd, the expected proportion of discriminators.
}
\usage{
discrimSS(pdA, pd0 = 0, target.power = 0.90, alpha = 0.05,
          pGuess = 1/2, test = c("difference", "similarity"),
          statistic = c("exact", "stable.exact", "both.exact",
           "normal", "cont.normal"))

d.primeSS(d.primeA, d.prime0 = 0, target.power = 0.90, alpha = 0.05,
          method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
            "triangle"),
          test = c("difference", "similarity"),
          statistic = c("exact", "stable.exact", "both.exact",
           "normal", "cont.normal"))
}
\arguments{
  \item{pdA}{the probability of discrimination for the model under the
    alternative hypothesis; scalar between zero and one}
  \item{d.primeA}{d-prime for the model under the
    alternative hypothesis; non-negative numerical scalar}
  \item{pd0}{the probability of discrimination under the
    null hypothesis; scalar between zero and one}
  \item{d.prime0}{d-prime under the null hypothesis; non-negative
    numerical scalar}
  \item{target.power}{the desired power for the test}
  \item{alpha}{the type I level of the test; scalar between zero and
    one}
  \item{method}{the discrimination protocol for which the sample size
    should be computed}
  \item{pGuess}{the guessing probability for the discrimination
    protocol, e.g. 1/2 for duo-trio and 2-AFC, and 1/3 for
    triangle, tetrad and 3-AFC; scalar between zero and one}
  \item{test}{the type of one-sided binomial test (direction of the
    alternative hypothesis): "difference" corresponds "greater" and
    "similarity" corresponds to "less"}
  \item{statistic}{options are explained in the Details section below}
}
\details{
  For difference tests \code{pdA} or \code{d.primeA} (the sensory
  difference under the alternative hypothesis) has to be larger than
  \code{pd0} or \code{d.prime0} (the sensory  difference under the null
  hypothesis). The sample size of the standard one-tailed difference
  test where the null hypothesis of "no difference" is obtained with
  \code{pd0 = 0} or \code{d.prime0 = 0}.

  For similarity tests it is required that \code{pd0} > \code{pdA} or
  equivalently that \code{d.prime0} > \code{d.primeA}. Here, the
  interval [0, \code{pdA}] or [0, \code{d.primeA}]
  is the similarity region covering sensory differences for which we
  would say that the products are similar.

  The probability of a correct answer under the null hypothesis is
  given by \code{pd0 + pGuess * (1 - pd0)}. Similarly, the probability
  of a correct answer under the alternative hypothesis is given by
  \code{pdA + pGuess * (1 - pdA)}.

  The \code{statistic} argument:
  \itemize{
    \item{\code{"exact"}}{
      is the conventional sample size for the exact binomial test:
      The smallest sample size that gives the desired power
      (\code{target.power}) at the given
      significance level. Ususally slightly higher sample sizes will not
      have the desired power, however. This is due to the non-monotonic
      behavior of power as a function of sample size.
    }
    \item{\code{"stable.exact"}}{
      is so-called stable exact sample size proposed by Ennis and
      Jesionka (2011) which has the property that no larger sample sizes
      has a power less than the \code{target.power}.
    }
    \item{\code{"both.exact"}}{
      returns both \code{exact} and \code{stable.exact}
      sample sizes
    }
    \item{\code{"normal"}}{
      is the normal approximation to the exact binomial sample size
      without any continuity adjustment. This usually provides a sample
      size that is smaller than the sample size for the exact binomial
      test.
    }
    \item{\code{"cont.normal"}}{
      is the continuity adjusted normal approximation to
      the sample size for the exact binomial test. This sample size is
      usually closer to the exact sample size than the unadjusted
      approximation and usually higher than the unadjusted
      approximation.
    }
  }

  If the sample size based on the continuity adjusted normal
  approximation is larger than 10,000, the function returns the normal
  approximation and issues a warning.
}
\value{
  The sample size; an integer.
}
\references{
  Brockhoff, P.B. and Christensen, R.H.B (2010). Thurstonian
  models for sensory discrimination tests as generalized linear models.
  Food Quality and Preference, 21, pp. 330-338.

  Ennis, J.M. and V. Jesionka (2011). The power of sensory
  discrimination methods revisited. Journal of Sensory Studies, 26,
  pp. 371-382.
}
\author{Per Bruun Brockhoff and Rune Haubo B Christensen}

\seealso{\code{\link{AnotA}}, \code{\link{discrimPwr}},
  \code{\link{samediff}}, \code{\link{findcr}}}

\examples{
## Finding the smallest necessary sample size:
discrimSS(pdA = 0.5, pd0 = 0, target.power = 0.80, alpha = 0.05,
   pGuess = 1/2, test = "difference", statistic = "exact")
## The stable-exact sample size is larger:
discrimSS(pdA = 0.5, pd0 = 0, target.power = 0.80, alpha = 0.05,
   pGuess = 1/2, test = "difference", statistic = "stable.exact")

## Give identical results:
pd <- coef(rescale(d.prime = 1, method = "twoAFC"))$pd
discrimSS(pdA = pd, pd0 = 0, target.power = 0.90, alpha = 0.05,
   pGuess = 1/2, test = "difference", statistic = "exact")
d.primeSS(1, target.power = 0.90, method = "twoAFC")

## A similarity example:
discrimSS(pdA = 0.1, pd0 = 0.2, target.power = 0.80, alpha = 0.05,
   pGuess = 1/2, test = "similarity", statistic = "exact")
\dontshow{
## Test stability of code:
## Finding the smallest necessary sample size:
a <- discrimSS(pdA = 0.5, pd0 = 0, target.power = 0.80, alpha = 0.05,
   pGuess = 1/2, test = "difference", statistic = "exact")
a2 <- discrimSS(pdA = 0.5, pd0 = 0, target.power = 0.80, alpha = 0.05,
   pGuess = 1/2, test = "difference", statistic = "stable.exact")

## Give identical results:
pd <- coef(rescale(d.prime = 1, method = "twoAFC"))$pd
b <- discrimSS(pdA = pd, pd0 = 0, target.power = 0.90, alpha = 0.05,
   pGuess = 1/2, test = "difference", statistic = "exact")
c <- d.primeSS(1, target.power = 0.90, method = "twoAFC")

## A similarity example:
d <- discrimSS(pdA = 0.1, pd0 = 0.2, target.power = 0.80, alpha = 0.05,
   pGuess = 1/2, test = "similarity", statistic = "exact")

expect <- c(23, 28, 30, 30, 604)
stopifnot(c(a, a2, b, c, d) == expect)
}
}
\keyword{models}% at least one, from doc/KEYWORDS
