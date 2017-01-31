\name{profile.discrim}
\alias{profile.discrim}
\alias{plot.profile.discrim}
\alias{confint.discrim}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Profile likelihood and confidence interval methods for discrim
  objects} 
\description{
  Computes the (normalized or relative) profile likelihood for the
  parameters of a discrimination test, plots the normalized profile
  likelihood.
}
\usage{
\method{profile}{discrim}(fitted, ...)

\method{plot}{profile.discrim}(x, level = c(0.99, 0.95), fig = TRUE,
            method = "natural", n = 1e3, ...)

\method{confint}{discrim}(object, parm, level = 0.95, ...)
}
\arguments{
  \item{fitted}{a \code{discrim} object}
  \item{x}{a \code{profile.discrim} object}
  \item{object}{a \code{discrim} object}
  \item{parm}{currently not used}
  \item{method}{the type of spline to be used in approximating the
    profile likelhood curve (trace)---se \code{\link{spline}} for
    details}
  \item{n}{the number of spline interpolations to use in plotting the
    profile likelihood curve (trace)}
  \item{level}{for \code{plot}: At which levels to include horizontal lines to indicate
    confidence levels in plots of the normalized profile
    likelihoods. For \code{confint}: at which level to compute the
    confidence interval}
  \item{fig}{logical: should the normalized profile likelihoods be plotted?}
  \item{\dots}{For \code{plot}: additional arguments to
    \code{plot}. Otherwise not used.}
}
\details{
  \code{confint} returns the confidence interval computed in
  \code{\link{discrim}} possibly at another level. The statistic used to
  compute the confidence interval is therefore determined in the
  \code{\link{discrim}} call and may not be the likelihood root.

  The likelihood profile is extracted from the \code{\link{discrim}}
  object fitted with \code{statistic = "likelihood"}.
}
\value{
  For \code{profile}:
  An object of class \code{"profile.discrim", "data.frame"}---a
  \code{data.frame} with two columns giving
  the value of the parameter and the corresponding value of the profile
  likelihood.

  For \code{plot}:
  The profile object is returned invisibly.

  For \code{confint}:

  A 3x2 matrix with columns named \code{"lower", "upper"} giving the
  lower and upper (100 * \code{level})\% confidence interval for the
  parameters named in the rows.
}
\references{Brockhoff, P.B. and Christensen R.H.B. (2010). Thurstonian
  models for sensory discrimination tests as generalized linear models.
  Food Quality and Preference, 21, pp. 330-338.}
\author{Rune Haubo B Christensen and Per Bruun Brockhoff}

\seealso{
  \code{\link{discrim}}
}

\examples{
## 7 success out of 10 samples in a duo-trio experiment:
(dd <- discrim(7, 10, method = "duotrio", statistic = "likelihood"))
confint(dd)
plot(profile(dd))
points(confint(dd)[3,], rep(.1465, 2), pch = 3, cex = 2, lwd=2)

}
\keyword{models}% at least one, from doc/KEYWORDS
