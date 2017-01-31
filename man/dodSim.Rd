\name{dodSim}
\alias{dodSim}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Simulate data from the Degree-of-Difference model}
\description{
  Simulate data from the Degree-of-Difference model for a given value of
  d-prime. The boundary parameters can either be specified by the user,
  or be chosen automatically so as to 1) maximize the likelihood ratio
  statistic, 2) ensure responses in each category is equally probable
  across same-pairs and different-pairs or 3) minimize the standard error of
  d-prime.
}
\usage{

dodSim(d.prime, ncat=4, sample.size = c(100, 100),
       method.tau = c("equi.prob", "LR.max", "se.min", "user.defined"),
       tau = NULL, d.prime0 = 0, ...)

}
\arguments{
  \item{d.prime}{the value of d-prime.}
  \item{ncat}{the number of response categories.}
  \item{sample.size}{the sample size for same-pairs and
    different-pairs. The sample size can be a scalar number in which
    case the sample sizes for both same-pairs and different-pairs are taken to
    equal that number.}
  \item{method.tau}{the method with which to choose the boundary
    parameters. If \code{"user.defined"}, the user has to specify the
    \code{tau} argument, otherwise the set of boundary parameters are
    chosen automatically (see the Details section below).}
  \item{tau}{if \code{method.tau = "user.defined"} the set of boundary
    parameters, otherwise not used.}
  \item{d.prime0}{if \code{method.tau = "LR.max"} the value of d-prime
    under the null hypothesis, otherwise not used.}
  \item{\dots}{passed on to \code{\link{optimal_tau}}.}
}
\details{
  In principle both d-prime and all boundary parameters have to be
  specified in order to be able to simulate from the DOD model.
  However, since it
  can be difficult to decide which boundary parameters to use for
  simulation, \code{dodSim} offers ways to choose these parameters
  automatically according to the following three criteria:
  \describe{
    \item{equi.prob}{the boundary parameters are chosen such that
      responses in each category are equally probable across same-pairs
      and different-pairs.}
    \item{LR.max}{the boundary parameters are chosen such that the
      likelihood ratio statistic for the test of d-prime is
      maximized. This choice maximizes the power of the likelihood ratio
      test and is in a sense an optimal choice of boundary parameters.}
    \item{se.min}{the boundary parameters are chosen such that the
      standard error of d-prime is minimized. This method also maximizes
      the power of the Wald test of d-prime when the null hypothesis is
      no-difference (d-prime = 0). This method can be numerical unstable
      for small and large d-prime values (approximately d.prime < 0.5 and
      d.prime > 5).}
  }

  Experience shows that the asymptotic properties of the DOD model are
  not too sensitive to the choice of boundary parameters: power,
  standard error of d-prime and confidence intervals seem to be fairly
  constant irrespectively which of the above three criteria are used to
  choose the boundary parameters.
}
\value{
  A 2-by-\code{ncat} matrix of counts with same-pairs in the first
  row and different-pairs in the second row. First/last column
  corresponds to "same"/"different" on the response scale.
}
%% \references{
%%   Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian
%%   models for sensory discrimination tests as generalized linear models.
%%   Food Quality and Preference, 21, pp. 330-338.
%% }
\author{Rune Haubo B Christensen}

\seealso{ \code{\link{dod}}, \code{\link{dod_fit}},
  \code{\link{dodControl}}, \code{\link{optimal_tau}},
  \code{\link{dodPwr}}
}

\examples{

## Simulate data from the DOD model with the equi.prob method:
set.seed(125)
(Data <- dodSim(d.prime=1, sample.size=100, method.tau="equi.prob"))

## Simulate data that maximizes the likelihood ratio statistic:
set.seed(125)
dodSim(d.prime=1, sample.size=100, method.tau="LR.max")

## Simulate with user-defined values for the boundary parameters:
dodSim(d.prime=1.5, sample.size=c(100, 100),
       method.tau="user.defined", tau=1:4)

## Simulate using different sample sizes for same-pairs and
## different-pairs:
dodSim(d.prime=1, ncat=3, sample.size=c(75, 125),
       method.tau="se.min")

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
