\name{AUC}
\alias{AUC.default}
\alias{AUC.anota}
\alias{print.AUC}
\alias{AUC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ AUC computation}
\description{
  This is the default AUC function for scalar d-primes, which will
  compute Area Under the ROC curve (ROC is an acronym for receiver
  operating characteristic) assuming a  normal distribution
  for the underlying percepts.
}
\usage{

\method{AUC}{default}(d, se.d, scale = 1, CI.alpha = 0.05, ...)

\method{AUC}{anota}(d, CI.alpha = 0.05, ...)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{d}{a unit lenght vector with the value of d-prime for which AUC
    is to be computed or a \code{anota} object from the fitting of a
    A-not A test with \code{\link{AnotA}}}
  \item{scale}{a unit length vector giving the ratio of scale (ie. standard
    deviation) of the latent distribution for the no-class items
    relative to that of the yes-class items}
  \item{se.d}{standard error of \code{d} (d-prime). If provided, the
    function will compute confidence limits of value of AUC---cf. in
    section value.}
  \item{CI.alpha}{the type I level of the confidence interval of AUC}
  \item{\dots}{additional arguments passed \code{integrate}}
}
\details{
  The AUC is computed using the standard normal distribution function
  \code{\link{pnorm}}.

  Confidence limits are based on a normal approximation of
  \code{d} and not of AUC. The limits are computed,
  if an estimate of the standard error of \code{d} is provided. Note
  that the limits do not take the uncertainty in estimating the scale
  nor that of estimating the standard error of \code{d} into account.

  A print method is implemented for objects of class \code{AUC}.
}
\value{
  A list with components. If \code{se.d} is supplied to the default
  method or if a discrim object is supplied, the object contains the
  latter three additional elements.

  \item{value}{the estimated value of AUC}
  \item{res.int}{the result from the call to \code{integrate}}
  \item{lower}{the lower confidence limit}
  \item{upper}{the upper confidence limit}
  \item{CI.alpha}{echoes the provided \code{CI.alpha}}
}
%% \references{}
\author{Rune Haubo B Christensen}
%%\note{ ~~further notes~~
%%
%% ~Make other sections like Warning with \section{Warning }{....} ~
%%}
%%\seealso{ ~~objects to See Also as \code{\link{help}}, ~~~ }
\examples{

## Compute AUC from d-prime and confindence interval for the AUC:
fm1 <- AnotA(8, 25, 1, 25)
AUC(d=fm1$coef, se.d=fm1$se)
## The AUC-method for AnotA-objects can be used for convenience:
AUC(fm1)

\dontshow{
## Test agreement of AUC computations:
x <- AUC(fm1$coef, fm1$se)
y <- AUC(fm1)
z <- pnorm(fm1$coef/sqrt(2))
## w := dput(as.vector(z))
w <- 0.817851857437929

stopifnot(
  isTRUE(all.equal(as.vector(x$value), as.vector(y$value),
                   as.vector(z), w)),
  isTRUE(all.equal(x$lower, y$lower)),
  isTRUE(all.equal(x$upper, y$upper))
)

## Don't actually know what this is supposed to illustrate:
(odor <- matrix(c(112, 112, 72, 53, 22, 4, 7, 38, 50, 117, 101, 62), 2,
               byrow = TRUE))
(d.primes <- SDT(odor)[,3])
for(i in 1:5) print(AUC(d.primes[i]))
}

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{htest}
%% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
