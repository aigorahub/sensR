\name{dodControl}
\alias{dodControl}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Control settings for the dod function}
\description{
  Specify control setting when fitting the the Thurstonian
  Degree-of-Difference (DOD) model using \code{\link{dod}} and
  \code{\link{dod_fit}}.
}
\usage{

dodControl(grad.tol = 1e-4,
           integer.tol = 1e-8,
           get.vcov = TRUE,
           get.grad = TRUE,
           test.args = TRUE,
           do.warn=TRUE,
           optCtrl=list())


}
\arguments{
  \item{grad.tol}{tolerance for the maximum absolute gradient of the
    parameters are convergence.}
  \item{integer.tol}{tolerance for when to give a warning about
    non-integer counts in data.}
  \item{get.vcov}{compute the variance-covariance matrix of the
    parameters (and the standard error of d-prime)?}
  \item{get.grad}{compute the gradient of the parameters?}
  \item{test.args}{test admissibility of arguments to \code{\link{dod}}
    and \code{\link{dod_fit}}?}
  \item{do.warn}{if \code{FALSE} warnings from the fitting process are
    suppressed.}
  \item{optCtrl}{control parameters passed on to the
    \code{\link{nlminb}} optimizer.}
}
\value{
  An list of class \code{dodControl} with the appropriate control
  settings for \code{\link{dod}} and \code{\link{dod_fit}}.
}

\author{Rune Haubo B Christensen}

\seealso{ \code{\link{dod}}, \code{\link{dod_fit}},
  \code{\link{dodSim}},
  \code{\link{dodPwr}}, \code{\link{optimal_tau}},
}

\examples{

## DOD example data:
same.pairs <- c(25, 22, 33, 20)
diff.pairs <- c(18, 22, 30, 30)

## Fit Thurstonian dod-model and perform difference test:
dod(same=same.pairs, diff=diff.pairs)

## Display the fitting process using the trace argument to nlminb:
ctrl <- dodControl(optCtrl=list(trace=TRUE))
dod(same=same.pairs, diff=diff.pairs, control=ctrl)

\dontshow{

}
}
\keyword{models}% at least one, from doc/KEYWORDS
