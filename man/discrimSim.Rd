\name{discrimSim}
\alias{discrimSim}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Simulates replicated difference tests}
\description{
  Simulates the outcome of \code{sample.size} replicated sensory
  difference tests (for any one of four protocols: 2-AFC, 3-AFC,
  duotrio, tetrad and triangle tests)
  for a given d-prime value and a given overdispersion (default 0).
}
\usage{
discrimSim(sample.size, replicates, d.prime, sd.indiv = 0,
           method = c("duotrio", "halfprobit", "probit", "tetrad",
             "triangle", "twoAFC", "threeAFC"))
}
\arguments{
  \item{sample.size}{the sample size - number of subjects}
  \item{replicates}{number of replications per subject}
  \item{d.prime}{the value of d-prime}
  \item{method}{the discrimination protocol}
  \item{sd.indiv}{the individual variability in d-prime values. A value
    of 0 (default) corresponds to complete independence}
}
\details{
  The d-prime for each subject is a random draw from a normal
  distribution with mean \code{d.prime} and standard deviation
  \code{sd.indiv}. All negative values are set to zero.
}
\value{
  A vector of length \code{sample.size} with the number of correct
  answers for each subject.
}
\references{
  Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian
  models for sensory discrimination tests as generalized linear models.
  Food Quality and Preference, 21, pp. 330-338.
}
\author{Rune Haubo B Christensen and Per Bruun Brockhoff}

\seealso{ \code{\link{triangle}}, \code{\link{twoAFC}},
  \code{\link{threeAFC}}, \code{\link{duotrio}}, \code{\link{tetrad}},
  \code{\link{discrimPwr}}, \code{\link{discrim}},
  \code{\link{AnotA}}, \code{\link{discrimSS}},
  \code{\link{samediff}}, \code{\link{findcr}}}

\examples{

## Running simulations:
discrimSim(sample.size = 10, replicates = 3, d.prime = 2,
           method = "triangle", sd.indiv = 1)
\dontshow{
set.seed(1)
a <- discrimSim(sample.size = 10, replicates = 3, d.prime = 2,
           method = "triangle", sd.indiv = 1)
expect <- c(3L, 3L, 2L, 3L, 3L, 1L, 3L, 3L, 1L, 2L)
stopifnot(all(a == expect))
}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{models}
