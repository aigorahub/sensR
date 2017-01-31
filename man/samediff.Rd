\name{samediff}
\alias{samediff}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Computation of tau and dprime for same different test}
\description{
  Computation of tau and dprime and their uncertainties for the
  same different test using maximum likelihood.
}
\usage{
samediff(nsamesame, ndiffsame, nsamediff, ndiffdiff, VCOV = TRUE)
}
\arguments{
  \item{nsamesame}{The number of same-answers on same-samples}
  \item{ndiffsame}{The number of different-answers on same-samples}
  \item{nsamediff}{The number of same-answers on different-samples}
  \item{ndiffdiff}{ The number of different-answers on
    different-samples}
  \item{VCOV}{Should the variance-covariance matrix of the parameters be
  computed. Defaults to \code{TRUE}.}
}
\details{
  The function computes the maximum likelihood estimates of
  \code{tau} and \code{delta}.
}
\value{
  An object of class \code{samediff} with elements
  \item{coef}{ named vector of coefficients (d-prime and tau)}
  \item{vcov}{variance-covariance matrix of the coefficients}
  \item{se}{named vector with standard error of the coefficients
    (standard error of d-prime)}
  \item{data}{a named vector with the data supplied to the function}
  \item{test}{a string with the name of the test (\code{same-different})}
  \item{call}{the matched call}
  \item{convergence}{convergence indicater. 0 indicates convergence. For error
    codes see \code{\link{optim}}.}
  \item{logLik}{Value of the log-likelhood at the MLE of the
    parameters.}
  \item{case}{A case indicator for internal use}
}
\references{Christensen, R.H.B., Brockhoff, P.B. (2009).
  Estimation and inference in the same-different test.
  Food, Quality and Preference, 20 pp. 514--520}
\author{Rune Haubo B Christensen}

\examples{
# data: 8 of the same samples were judged to be same
#       5 of the same samples were judged to be different
#       4 of the different samples were judged to be same
#       9 of the different samples were judged to be different

samediff(8, 5, 4, 9)
}
\keyword{models}% at least one, from doc/KEYWORDS
