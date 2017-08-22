\name{discrimPwr}
\alias{discrimPwr}
\alias{d.primePwr}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Sensory discrimination power analysis}
\description{
  Computes the power of a difference or similarity test for a sensory
  discrimination experiment using the binomial distribution.
  \code{d.primePwr} is a convenience function that calls
  \code{discrimPwr} but has arguments in terms of d-prime rather than
  pd, the probability of discrimination.
}
\usage{
discrimPwr(pdA, pd0 = 0, sample.size, alpha = 0.05, pGuess = 1/2,
           test = c("difference", "similarity"),
           statistic = c("exact", "normal", "cont.normal"))

d.primePwr(d.primeA, d.prime0 = 0, sample.size, alpha = 0.05,
           method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
             "triangle", "hexad", "twofive", "twofiveF"),
           double = FALSE,
           test = c("difference", "similarity"),
           statistic = c("exact", "normal", "cont.normal"))
}
\arguments{
  \item{pdA}{the probability of discrimination for the model under the
    alternative hypothesis; scalar between zero and one}
  \item{d.primeA}{d-prime for the model under the
    alternative hypothesis; non-negative numerical scalar}
  \item{pd0}{the probability of discrimination under the
    null hypothesis; scalar between zero and one}
  \item{d.prime0}{d-prime under the null hypothesis; non-negative
    numerical scalar}
  \item{sample.size}{the sample size; a scalar positive integer}
  \item{alpha}{the type I level of the test; scalar between zero and
    one}
  \item{method}{the discrimination protocol for which the power should
    be computed}
  \item{double}{should the 'double' variant of the discrimination protocol
    be used? Logical scalar. Currently not implemented for "twofive",
    "twofiveF", and "hexad".}
  \item{pGuess}{the guessing probability for the discrimination
    protocol, e.g. 1/2 for duo-trio and 2-AFC, 1/3 for
    triangle, tetrad and 3-AFC, 1/10 for two-out-of-five and hexad
    and 2/5 for two-out-of-five with forgiveness; scalar between zero and one}
  \item{test}{the type of one-sided binomial test (direction of the
    alternative hypothesis): "difference" corresponds "greater" and
    "similarity" corresponds to "less"}
  \item{statistic}{should power determination be based on the 'exact'
           binomial test, the normal approximation to this, or the
           normal approximation with continuity correction?}
}
\details{
  The power of the standard one-tailed difference test where the null
  hypothesis is "no difference" is obtained with \code{pd0 = 0}.

  The probability under the null hypothesis is
  given by \code{pd0 + pg * (1 - pd0)} where \code{pg} is the guessing
  probability \code{pGuess}. Similarly, the probability of the
  alternative hypothesis is given by \code{pdA + pg * (1 - pdA)}
}
\value{
  The power; a numerical scalar.
}
\references{
  Brockhoff, P.B. and Christensen, R.H.B (2010). Thurstonian
  models for sensory discrimination tests as generalized linear models.
  Food Quality and Preference, 21, pp. 330-338.
  
  Bi, J. (2001) The double discrimination methods. Food Quality and
  Preference, 12, pp. 507-513.
}
\author{Rune Haubo B Christensen and Per Bruun Brockhoff}

\seealso{\code{\link{findcr}},
  \code{\link{discrim}}, \code{\link{discrimSim}},
  \code{\link{AnotA}}, \code{\link{discrimSS}}
}

\examples{
## Finding the power of a discrimination test with d-prime = 1,
## a sample of size 30 and a type I level of .05:
pd <- coef(rescale(d.prime = 1, method = "twoAFC"))$pd
discrimPwr(pd, sample.size = 30)
d.primePwr(1, sample.size = 30, method = "twoAFC")
## Obtaining the equivalent normal approximation with and without
## continuity correction:
discrimPwr(pd, sample.size = 30, statistic = "cont.normal")
discrimPwr(pd, sample.size = 30, statistic = "normal")

# Example from Bi (2001) with n=100 and 35 correct answers in a 
# double duotrio test:
p1 <- 0.35
# Estimate of d-prime quoted by Bi(2001) was 1.06:
dp <- psyinv(p1, method="duotrio", double=TRUE) 
# Power using normal approximation w/o continuity adjustment quoted by Bi(2001):
d.primePwr(dp, sample.size = 100, method="duotrio", 
           double=TRUE, stat="normal") # 0.73
# d.primePwr(dp, sample.size = 100, method="duotrio", double=TRUE, 
#            stat="cont.normal")

# Power of exact test:
d.primePwr(dp, sample.size = 100, method="duotrio", 
           double=TRUE, stat="exact") # 0.697

## A similarity example:
discrimPwr(pdA = 0.1, pd0 = 0.2, sample.size = 100, pGuess = 1/3,
           test = "similarity")
\dontshow{
## Testing stability of results:
x1 <- discrimPwr(pd, sample.size = 30)
x2 <- d.primePwr(1, sample.size = 30, method = "twoAFC")
## x3 := dput(x2)
x3 <- 0.917255217993622
stopifnot(isTRUE(all.equal(x1, x2, x3)))

y1 <- discrimPwr(pd, sample.size = 30, statistic = "cont.normal")
y2 <- discrimPwr(pd, sample.size = 30, statistic = "normal")
y3 <- discrimPwr(pdA = 0.1, pd0 = 0.2, sample.size = 100, pGuess = 1/3,
           test = "similarity")
y <- c(y1, y2, y3)
## yres := dput(y)
yres <- c(0.88466478871626, 0.921092040377824, 0.306809762270984)
stopifnot(isTRUE(all.equal(y, yres)))

# Check examples from Bi (2001):
p1 <- 0.35
stopifnot(
  all.equal(psyinv(p1, method="duotrio", double=TRUE), 1.06082607037307),
  all.equal(d.primePwr(dp, sample.size = 100, method="duotrio", 
                       double=TRUE, stat="normal"), 0.726811262989353),
  all.equal(d.primePwr(dp, sample.size = 100, method="duotrio", 
           double=TRUE, stat="exact"), 0.697083711530984))
}

}
\keyword{models}% at least one, from doc/KEYWORDS
