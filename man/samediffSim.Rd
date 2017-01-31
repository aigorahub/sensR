\name{samediffSim}
\alias{samediffSim}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Simulates data from a samediff test}
\description{
  Simulates the outcome of \code{n} same-different experiments.
}
\usage{
samediffSim(n, tau, delta, Ns, Nd)
}
\arguments{
  \item{n}{the number of experiments to simulate.}
  \item{tau}{the value of "tau".}
  \item{delta}{the value of delta (d-prime).}
  \item{Ns}{number of same-samples}
  \item{Nd}{number of different-samples}
}
\details{
  The function makes two calls to \code{\link{rbinom}}.
}
\value{
  A matrix of with \code{n} rows and four columns named \code{ss, ds,
    sd, dd} with the number of same-answers to same-samples,
  different-answers to same-samples, same-answers to different-samples
  and different-answers to different-samples respectively.
}
\references{Christensen, R.H.B., Brockhoff, P.B. (2009).
  Estimation and inference in the same-different test.
  Food, Quality and Preference, 20 pp. 514--520}
\author{Rune Haubo B Christensen}

\seealso{ \code{\link{discrimSim}}}

\examples{
## Running simulations:
samediffSim(n = 10, tau = 1, delta = 1, Ns = 10, Nd = 10)

}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{models}
