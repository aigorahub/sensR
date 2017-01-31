\name{dod}
\alias{dod}
\alias{print.dod}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Thurstonian Degree-of-Difference (DOD) model}
\description{
  Fits the Thurstonian Degree-of-Difference (DOD) model and performs
  hypothesis/significance tests of d-prime (Thurstonian
  delta). One-sided difference and similarity tests as well as two-sided
  tests of d-prime are available. The user may choose from a number of
  tests statistics.
}
\usage{

dod(same, diff, d.prime0 = 0, conf.level = 0.95,
    statistic = c("likelihood", "Pearson", "Wilcoxon", "Wald"),
    alternative = c("difference", "similarity", "two.sided",
    "less", "greater"), control=dodControl(), ...)

\method{print}{dod}(x, digits = max(3, getOption("digits") - 3), ...)

}
\arguments{
  \item{same}{the answers to same-pairs; either 1) a numeric vector of
    counts of length equal to the number of response categories
    ordered appropriately or 2) a
    factor where the levels indicate the response categories.}
  \item{diff}{the answers to different-pairs in the same format as
    \code{same}.}
  \item{d.prime0}{the value of d.prime under the null hypothesis. In the
        standard no-difference test \code{d.prime0 = 0}, while it has to
        be positive for similarity tests and two-sided tests.}
  \item{conf.level}{the confidence level for the confidence intervals}
  \item{statistic}{the statistic to be used for hypothesis testing}
  \item{alternative}{the nature of the alternative hypothesis in the
        hypothesis/significance test for d-prime. Note that
	\code{"greater"} is an alias for \code{"difference"} and
	\code{"less"} is an alias for \code{"similarity"}}
  \item{control}{options to control the fitting process specfied via a
    call to \code{\link{dodControl}}.}
  \item{x}{an object of class \code{"dod"}.}
  \item{digits}{number of digits in resulting table of results.}
  \item{\dots}{not currently used.}
}
\details{
  \code{dod} will report the likelihood based confidence interval for
  d.prime unless \code{statistic = "Wald"} in which case the
  standard symmetric Wald type confidence interval is
  reported. This interval can be highly inaccurate and so is not
  recommened for practical use.

  The p-value for the standard one-tailed difference test of "no
  difference" is obtained with \code{d.prime0 = 0} corresponding to the
  default setting.

  The standard error of d-prime is not defined when the parameter
  estimate is zero (or numerically close) and it will be reported as
  \code{NA} in this case.

  The \code{"Wald"} statistic is *NOT* recommended for practical
  use---it is only included here for completeness and to allow
  comparisons with other software etc.

}
\value{
  An object of class \code{dod}.

  %% with elements
  %% \item{coefficients}{matrix of estimates, standard errors and
  %%   confidence intervals}
  %% \item{data}{a named vector with the data supplied to the function}
  %% \item{p.value}{the p-value of the hypothesis test}
  %% \item{call}{the matched call}
  %% \item{test}{the type of test}
  %% \item{statistic}{the statistic used for confidence intervals and
  %%   p-value}
  %% \item{d.prime0}{the value of d-prime under the
  %%   null hypothesis}
  %% \item{conf.level}{the confidence level}
  %% \item{stat.value}{value of the
  %%   test statistic used to calculate the p-value}
}

\references{
  Ennis, J.M. and R.H.B. Christensen (2014) A Thurstonian comparison
  of the tetrad and degree of difference tests.
  \emph{Food Quality and Preference}, in press.

  Christensen, R.H.B, J.M. Ennis, D.M. Ennis and P.B Brockhoff (2012)
  A Thurstonian model for the Degree of Difference test
  with extensions to unequal variance, sequence effects
  and replicated data. Talk at Sensometrics conference, Rennes, France,
  July 11th.
}

\author{Rune Haubo B Christensen}

\seealso{
  \code{\link{dodSim}}, %% \code{\link{optimal_tau}},
  \code{\link{dodPwr}},
  \code{\link{dodControl}}, \code{\link{dod_fit}},
  \code{\link{optimal_tau}}
}

\examples{

## DOD example data:
same.pairs <- c(25, 22, 33, 20)
diff.pairs <- c(18, 22, 30, 30)

## Fit Thurstonian dod-model and perform difference test:
dod(same=same.pairs, diff=diff.pairs)

## Can choose another test statistic (e.g. Wilcoxon):
dod(same=same.pairs, diff=diff.pairs, statistic="Wilcox")

## A similarity test (with simulated data):
set.seed(121)
(Data2 <- dodSim(d.prime=0, ncat=4, sample.size=200, method.tau="equi.prob"))
dod(same=Data2[1, ], diff=Data2[2, ], d.prime0=1.2,
    alternative="similarity")

## Extract parameters from a dod fit:
fm <- dod(same=same.pairs, diff=diff.pairs)
coef(fm)

\dontshow{

}
}
\keyword{models}% at least one, from doc/KEYWORDS
