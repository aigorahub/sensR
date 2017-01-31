\name{dodPwr}
\alias{dodPwr}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Power of the Degree-of-Difference (DOD) method}
\description{
  Computes the power of the Degree-of-Difference (DOD) method by
  simulation
}
\usage{

dodPwr(d.primeA, d.prime0=0, ncat = 4, sample.size, nsim = 1e3,
       alpha = 0.05,
       method.tau=c("LR.max", "equi.prob", "se.min", "user.defined"),
       statistic=c("likelihood", "Wilcoxon", "Pearson", "Wald"),
       alternative = c("difference", "similarity", "two.sided",
       "less", "greater"),
       tau=NULL, ...)

}
\arguments{
  \item{d.primeA}{the value of d-prime under the alternative hypothesis;
    non-negative numerical scalar.}
  \item{d.prime0}{the value of d-prime under the null hypothesis.}
  \item{ncat}{the number of response categories in the DOD model}
  \item{sample.size}{the sample size in each simulation for each of the
    same-pairs and different pairs. Can be a single scalar value or a
    2-vector.}
  \item{nsim}{the number of simulations.}
  \item{alpha}{the significance level.}
  \item{method.tau}{the method with which to choose the boundary
    parameters - see \code{\link{dodSim}} for details on the methods.}
  \item{statistic}{the statistic to be used for hypothesis testing.}
  \item{alternative}{the nature of the alternative hypothesis in the
        hypothesis/significance test for d-prime. Note that
	\code{"greater"} is an alias for \code{"difference"} and
	\code{"less"} is an alias for \code{"similarity"}.}
  \item{tau}{if \code{method.tau = "user.defined"} a vector of boundary
    parameters in the DOD model, otherwise not used.}
  \item{\dots}{parsed on to \code{\link{wilcox.test}} when appropriate.}
}
%% \details{
%% }
\value{
  The simulation based estimate of the power with the following
  attributes:
  \item{se(power)}{the estimated standard error of the estimated
    power. This is based on the formula
    \code{sqrt(pow * (1 - pow) / n)}, where \code{pow} is the estimated
    power and \code{n} is the number of simulations used to estimate the
    power.}
  \item{n.used}{the number of simulations used to estimate the
    power. This is usually equal to nsim, but can sometimes be smaller
    than nsim due to non-convergences to which the Wald test is
    especially prone.}
}

\references{
  Ennis, J.M. and R.H.B. Christensen (2014) A Thurstonian comparison
  of the tetrad and degree of difference tests.
  \emph{Food Quality and Preference}, in press.
}

\author{Rune Haubo B Christensen}

\seealso{ \code{\link{dod}}, \code{\link{dod_fit}},
  \code{\link{dodSim}}, \code{\link{optimal_tau}},
  \code{\link{dodControl}}
}

\examples{

## NOTE: The number of simulations (nsim) is set unrealistically low in
## the examples below to reduce the computation time for automatic
## package checks. nsim between 1e3 and 1e4 is usually sufficient and
## the latter often on the safe side. The standard error of the
## estimated power ('se(power)') reported by dodPwr() measures the
## accuracy of the estimated power and indicates if nsim needs to be
## increased.

## Estimate power of the conventional difference test (no-difference
## under the null hypothesis):
set.seed(125)
dodPwr(d.primeA=1, d.prime0=0, ncat=4, sample.size=100, nsim=50,
       alpha=.05, method.tau="LR.max", statistic="likelihood")
##  [1] 0.62
##  attr(,"se(power)")
##  [1] 0.1825346
##  attr(,"n.used")
##  [1] 50
## Here the boundary parameters are chosen automatically so as to
## maximize the likelihood ratio test statistic, and so this setting
## amounts to a highest achievable power scenario given d-prime = 1.

## Using another (and faster) statistic:
dodPwr(d.primeA=1, d.prime0=0, ncat=4, sample.size=100, nsim=1e3,
       alpha=.05, method.tau="LR.max", statistic="Wilcox")

\dontrun{
## Not automatically run to reduce computation time.

## Power of a similarity test:
set.seed(127)
dodPwr(d.primeA=0, d.prime0=1, ncat=4, sample.size=100, nsim=1e2,
       alpha=.05, method.tau="LR.max", statistic="Pearson",
       alternative="similarity")
##  [1] 0.71
##  attr(,"se(power)")
##  [1] 0.1434922
##  attr(,"n.used")
##  [1] 100

## Same as above, but with a given set of boundary parameters:
dodPwr(d.primeA=0, d.prime0=1, sample.size=100, nsim=1e2,
       alpha=.05, method.tau="user.defined", statistic="Pearson",
       alternative="similarity", tau=1:3)

## Using parallel computing to speed up computations:
if(require(parallel)) {
    ## Use detectCores() to get an appropriate number of cores for
    ## practical use - for the example here we fix it at 2:
    ## cl <- makeCluster(detectCores())
    cl <- makeCluster(getOption("cl.cores", 2))
    dvec <- c(0, .2, .5, .7, 1, 1.2, 1.5, 1.75)
    system.time(
        res <- parLapply(cl, dvec, fun=function(dp) {
            library(sensR)
            x <- dodPwr(dp, 0, sample.size=100, nsim=1e4, stat="Wil")
            c("power"=x, "se"=attr(x, "se(power)"))
        })
        )
    stopCluster(cl)
    names(res) <- dvec
    mat <- do.call(cbind, res)
    round(mat[1:2, ], 3)
    ## Example output:
    ##            0   0.2   0.5   0.7     1   1.5  1.75     2
    ##  power 0.051 0.058 0.123 0.238 0.578 0.983 1.000 1.000
    ##  se    0.022 0.023 0.033 0.043 0.049 0.013 0.002 0.001
}
}
## Realistically one should use more simulations, e.g. nsim=1e4.

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
