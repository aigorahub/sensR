\name{ROC}
\alias{ROC.default}
\alias{ROC.anota}
\alias{ROC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Plot the Receiver Operating Characteristic Curve}
\description{
  The function computes and plots the empirical ROC (receiver operating
  characteristic) curve.
}
\usage{
ROC(object, ...)

\method{ROC}{default}(object, se.d, scale = 1, length = 1000,
fig = TRUE, se.type = c("CI", "SE"), CI.alpha = 0.05, ...)

\method{ROC}{anota}(object, length = 1000, fig = TRUE,
se.type = c("CI", "SE"), CI.alpha = 0.05, ...) 
}
%- maybe also 'usage' for other objects documented here.
\arguments{
%%   \item{d}{a unit length vector with the value of d-prime for which the
%%     ROC curve should be computed and plotted}
%%   \item{x}{an object of class discrim fitted with \code{\link{AnotA}}}
  \item{object}{the class of the object defines, which of the methods is
    invoked. If obejct is a single element numeric vector it is taken as
    a d-prime value and the default method is invoked. If the object is
    of class \code{anota}, the method for \code{anota} objects is
    invoked.}  
  \item{se.d}{a unit length vector with the standard error of
    d-prime. If supplied confidence intervals or standard errors are
    plotted} 
  \item{scale}{a unit length vector giving the ratio of scale (ie. standard
    deviation) of the latent distribution for the no-class items
    relative to  that of the yes-class items} 
  \item{length}{the length of the vectors to be plotted. Longer vectors
    gives more smooth curves.}
  \item{fig}{Should a plot be produced?}
  \item{se.type}{The type of band for the ROC curve, \code{"CI"} for
    confidence interval and \code{"SE"} for standard error.}
  \item{CI.alpha}{the type I level of the confidence interval of AUC}
  \item{\dots}{additional arguments to \code{plot} and \code{lines}}
}
\details{
  The function currently ignores the variance of the scale in the
  computation of the uncertainty of the ROC curve.
}
\value{
  The function makes a plot of the ROC curve, and if \code{se.d} is
  supplied, standard errors or confidence intervals for the curve are
  added to the plot.

  The function also (invisibly) returns a list with the following
  components 
  \item{ROCx}{x-coordinates to the ROC curve}
  \item{ROCy}{y-coordinates to the ROC curve}

  If \code{se.d} is supplied, the object also contains
  \item{lower}{y-coordinates to the lower limit}
  \item{upper}{y-coordinates to the upper limit}
}
%% \references{ ~put references to the literature/web site here ~ }
\author{Rune Haubo B Christensen}
%% \note{ ~~further notes~~ 
%% 
%%  ~Make other sections like Warning with \section{Warning }{....} ~
%% }
%% \seealso{ ~~objects to See Also as \code{\link{help}}, ~~~ }
\examples{
## ROC.default:
(mat <- matrix(c(8, 17, 1, 24), 2, byrow = TRUE))
(d.prime <- SDT(mat, "probit")[3])
ROC(d.prime)
## ROC.anota:
fm1 <- AnotA(8, 25, 1, 25)
ROC(fm1)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{hplot}
%% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
