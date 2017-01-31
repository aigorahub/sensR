\name{dod_fit}
\alias{dod_fit}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Direct fitter of the Thurstonian Degree-of-Difference (DOD) model}
\description{
  Fits the Thurstonian Degree-of-Difference (DOD) model.
  This function is for programming use. The ordinary user probably wants
  the \code{\link{dod}} function, which is for interactive use.
  \code{dod_fit} only estimates the DOD model and performs no hypothesis
  or significance tests.
}
\usage{

dod_fit(same, diff, tau=NULL, d.prime=NULL, control=dodControl(),
        ...)

}
\arguments{
  \item{same}{the answers to same-pairs; either 1) a numeric vector of
    counts of length equal to the number of response categories
    ordered appropriately or 2) a
    factor where the levels indicate the response categories.}
  \item{diff}{the answers to different-pairs in the same format as
    \code{same}.}
  \item{tau}{optional vector of boundary parameters. If specified,
    \code{dod_fit} will not optimize over the \code{tau} parameters.}
  \item{d.prime}{optional d-prime value. If specified,
    \code{dod_fit} will not optimize over \code{d.prime}.}
  \item{control}{options to control the fitting process specfied via a
    call to \code{\link{dodControl}}.}
  \item{\dots}{not currently used.}
}
\details{
  The standard error of d-prime is not defined when the parameter
  estimate is zero (or numerically close) and it will be reported as
  \code{NA} in this case.
}
\value{
  An object of class \code{dod_fit}.

%%   with elements
%%   \item{coefficients}{matrix of estimates, standard errors and
%%     confidence intervals}
%%   \item{data}{a named vector with the data supplied to the function}
%%   \item{p.value}{the p-value of the hypothesis test}
%%   \item{call}{the matched call}
%%   \item{test}{the type of test}
%%   \item{statistic}{the statistic used for confidence intervals and
%%     p-value}
%%   \item{d.prime0}{the value of d-prime under the
%%     null hypothesis}
%%   \item{conf.level}{the confidence level}
%%   \item{stat.value}{value of the
%%     test statistic used to calculate the p-value}
}

\author{Rune Haubo B Christensen}

\seealso{
  \code{\link{dod}},
  \code{\link{dodSim}}, \code{\link{optimal_tau}},
  \code{\link{dodPwr}},
  \code{\link{dodControl}}
}

\examples{

## DOD example data:
same.pairs <- c(25, 22, 33, 20)
diff.pairs <- c(18, 22, 30, 30)

## Fit Thurstonian dod-model and perform difference test:
fm <- dod_fit(same=same.pairs, diff=diff.pairs)
names(fm)

## Estimate d-prime for given tau:
fm <- dod_fit(same=same.pairs, diff=diff.pairs, tau=1:3)

## Estimate tau for given d-prime:
fm <- dod_fit(same=same.pairs, diff=diff.pairs, d.prime=1)

\dontshow{

}
}
\keyword{models}% at least one, from doc/KEYWORDS
