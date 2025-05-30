\name{dprime_test}
\alias{dprime_test}
\title{
  Test of simple hypothesis with the common d-prime
}
\description{
  This function tests the hypothesis that the common d-prime is equal to
  or greater/less than a certain value, e.g. zero in a Wald or
  likelihood root test.
}
\usage{
dprime_test(correct, total, protocol, conf.level = 0.95, dprime0 = 0,
    statistic = c("likelihood", "Wald"),
    alternative = c("difference", "similarity", "two.sided", "less", "greater"),
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
    the confidence level for the confidence interval of the estimated
    common d-prime.
  }
  \item{dprime0}{
    Value of d-prime under the Null hypothesis. Non-negative numeric
    scalar.
  }
  \item{statistic}{
    the test statistic for computing the confidence interval as well as
    p-value.
  }
  \item{alternative}{
    the direction of the hypothesis test. \code{"difference"} and
    \code{"similarity"} are just alternative ways of specifying
    \code{"greater"} and \code{"less"} respectively.
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
  an object of class \code{"dprime_test"} with the following elements
  %% \item{stat.value}{
  %%   the value of the (chi-square) test statistic for the
  %%   'any-differences' hypothesis.
  %% }
  %% \item{df}{
  %%   the degrees of freedom for the \code{stat.value} test statistic.
  %% }
  \item{p.value}{
    the p-value for the 'any-differences' test.
  }
  \item{alternative}{
    character naming the direction of the hypothesis test.
  }
%% \item{res.dExp}{
%%   a list with \code{dExp}, the estimated common d-prime;
%%   \code{se.dExp}  standard error of the common and if \code{estim="ML"}
%%   \code{nll.dExp}
%% }
  \item{statistic}{
    the name of the test statistic.
  }
  \item{data}{
    the data table produced by \code{\link{dprime_table}}.
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
%% \references{
%% }
\author{
  Rune Haubo B Christensen
}
\seealso{
  \code{\link{dprime_compare}}, \code{\link{dprime_table}},
  \code{\link{posthoc.dprime_compare}}.
}
\examples{

n <- rep(40, 4)
x <- c(25, 25, 30, 35)
protocol <- c("triangle", "duotrio", "threeAFC", "twoAFC")
## Look at the data table with d-primes etc.:
dprime_table(x, n, protocol)

## Test of common d':
dprime_test(x, n, protocol)

## Another setting:
dprime_test(x, n, protocol, dprime0=2, statistic="Wald",
            alternative="less", estim="weighted.avg")

}
\keyword{models}
