\name{summary.samediff}
\alias{summary.samediff}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Summary method for samediff objects.}
\description{
  Makes a summary of a \code{samediff} object with option to use profile
  likelihood for confidence intervals and p-values or the assymptotic
  variance-covariance matrix.
}
\usage{
\method{summary}{samediff}(object, profile = TRUE, ...)
}
\arguments{
  \item{object}{a \code{samediff} object}
  \item{profile}{logical: Should the profile likelihood be used for
    confidence intervals and p-values for the parameters? Defaults to
    \code{TRUE}. If \code{FALSE} the assymptotic variance-covariance
    matrix derived from the observed Fisher information matrix will be
    used. See Details for more information.}
  \item{\dots}{can be \code{level}, eg 0.95 to specify the confidence
    level of the intervals.}
}
\details{
  Note that the variance-covariance matrix does not always exist in
  contrast to the profile likelihood. \code{profile = FALSE} may
  therefore cause confidence intervals etc. to be \code{NA}.
}
\value{
  An object of class \code{summary.samediff} inheriting elements from the
  \code{samediff} object and with the following additional elements
  \item{table}{matrix with parameter estimates, standard errors,
    confidence intervals and p-values.}
  \item{AIC}{the AIC of the object.}
}

\author{Rune Haubo B Christensen}

\seealso{
  \code{\link{confint.samediff}},   \code{\link{profile.samediff}}
}

\examples{
# data: 8 of the same samples were judged to be same
#       5 of the same samples were judged to be different
#       4 of the different samples were judged to be same
#       9 of the different samples were judged to be different

sadi <- samediff(8, 5, 4, 9)
summary(sadi)
summary(sadi, FALSE)

}
\keyword{models}% at least one, from doc/KEYWORDS
