\name{dprime_compare}
\alias{dprime_compare}
\title{
  Test the 'any-differences' hypothesis and estimate common d-prime
}
\description{
  This function will test the 'any-differences' hypothesis (conceptually
  a one-way ANOVA test for d-primes) with one of the Wald, Pearson or
  likelihood ratio chi-square test statistics. The common d-prime is
  estimated with ML or weighted average.
}
\usage{
dprime_compare(correct, total, protocol, conf.level = 0.95,
   statistic = c("likelihood", "Pearson", "Wald.p", "Wald.d"),
   estim = c("ML", "weighted.avg"))
}
\arguments{
  \item{correct}{
    a numeric vector of the number of correct answers; one element for
    each test.
  }
  \item{total}{
    a numeric vector of the total number of trials; one element for each
    test.
  }
  \item{protocol}{
    a character vector or factor naming the protocol used; one element
    for each test. Currently the following protocols are supported:
    \code{"triangle", "duotrio", "threeAFC", "twoAFC", "tetrad"}.
  }
  \item{conf.level}{
    the confidence level for the estimated common d-prime.
  }
  \item{statistic}{
    the test statistic for testing the 'any-differences' hypothesis.
  }
  \item{estim}{
    The estimation method for the common d-prime.
  }
}
\details{
  The vectors \code{correct}, \code{total} and \code{protocol} have to
  be of the same length.

  The function has a print method.
}
\value{
  an object of class \code{"dprime_compare"} with the following elements
  \item{stat.value}{
    the value of the (chi-square) test statistic for the
    'any-differences' hypothesis.
  }
  \item{df}{
    the degrees of freedom for the \code{stat.value} test statistic.
  }
  \item{p.value}{
    the p-value for the 'any-differences' test.
  }
  \item{statistic}{
    the name of the test statistic for the 'any-differences' test.
  }
  \item{data}{
    the data table produced by \code{\link{dprime_table}}.
  }
  \item{coefficients}{
    'table' with estimated common d-prime, standard error and confidence
    limits storred as a one-row \code{\link{data.frame}}.
  }
  \item{conf.level}{
    confidence level for the common d-prime.
  }
  \item{conf.int}{
    the confidence interval for the common d-prime.
  }
  \item{estim}{
    the estimation method for the common d-prime.
  }
  \item{conf.method}{
    the statistical method/test statistic used to compute the confidence
    interval for the common d-prime.
  }
}
\author{
  Rune Haubo B Christensen
}
\seealso{
  \code{\link{dprime_test}}, \code{\link{dprime_table}},
  \code{\link{posthoc.dprime_compare}}.
}
\examples{

## Make some fake data:
n <- rep(40, 4)
x <- c(25, 25, 30, 35)
protocol <- c("triangle", "duotrio", "threeAFC", "twoAFC")
## Look at the data table with d-primes etc.:
dprime_table(x, n, protocol)

## 'any differences' test:
## ML estimation and test with likelihood statistic:
(dpc <- dprime_compare(x, n, protocol))
## Other estimation/statistic options:
dprime_compare(x, n, protocol, estim="weighted.avg")
dprime_compare(x, n, protocol, statistic="Pearson")
dprime_compare(x, n, protocol, statistic="Wald.p")
dprime_compare(x, n, protocol, statistic="Wald.d")

}
\keyword{models}

