\name{findcr}
\alias{findcr}
\title{Find the critical value of a one-tailed binomial test}
\description{Finds the critical value in a one-tailed binomial test}
\usage{
findcr(sample.size, alpha = .05, p0 = .5, pd0 = 0,
              test = c("difference", "similarity"))
}
\arguments{
  \item{sample.size}{the sample size of the binomial test (must be a positve integer)}
  \item{alpha}{the test I error-level of the test (must be between zero and one)}
  \item{p0}{the guessing probability under the null-hypothesis (must be
    between zero and one); 1/2 for the duotrio and twoAFC tests and 1/3
    for the triangle, tetrad and threeAFC tests}
  \item{pd0}{the proportion of discriminators in the population of interest}
  \item{test}{the type of test}
}
\details{
  The critical value of the standard one-tailed difference test of "no
  difference" is obtained with \code{pd0 = 0}.

  The probability of a correct answer under the null hypothesis is
  given by \code{pd0 + p0 * (1 - pd0)}.
}
\value{
  The critical value in a one-tailed binomial test, that is, the smallest
  integer such that the null hypothesis binomial probability of being
  larger (smaller for similarity hypotheses) than or equal to this number
  is smaller than or equal to the type I error-level of the test.
}

\author{Rune Haubo B Christensen and Per Bruun Brockhoff}

\seealso{\code{\link{triangle}}, \code{\link{twoAFC}},
  \code{\link{threeAFC}}, \code{\link{duotrio}}, \code{\link{tetrad}},
  \code{\link{discrim}}, \code{\link{discrimPwr}},
  \code{\link{discrimSim}}, \code{\link{AnotA}}
  \code{\link{discrimSS}}, \code{\link{samediff}}}

\examples{
## Find the critical value for a triangle test for the level 0.05 test
## with 25 subjects:
findcr(sample.size = 25, , p0 = 1/3)

## Similarity example:
findcr(sample.size = 25, p0 = 1/3, pd0 = .2, test = "simil")
}
\keyword{models}% at least one, from doc/KEYWORDS

