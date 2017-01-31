\name{AnotA}
\alias{AnotA}
\alias{confint.anota}
\alias{plot.anota}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Analysis of A-not-A tests}
\description{
  Computation of dprime and it's uncertainty for the monadic A-not-A test
  together with the one-tailed P-value of the difference test
  (Fisher's Exact test).
}
\usage{
AnotA(x1, n1, x2, n2, ...)

\method{confint}{anota}(object, parm, level = 0.95, ...)

\method{plot}{anota}(x, main = TRUE, length = 1000, ...)

}
\arguments{
  \item{x1}{the number of (correct) A-answers on A-samples}
  \item{n1}{the total number of A-samples}
  \item{x2}{the number of A-answers on not-A-samples}
  \item{n2}{the number of not-A-samples}
  \item{object}{an \code{anota} object}
  \item{parm}{currently not used}
  \item{level}{the desired confidence level}
  \item{x}{an \code{anota} object}
  \item{main}{should the plot have a main title?}
  \item{length}{the discretization of the curves}
  \item{...}{additional arguments passed to \code{glm} for \code{AnotA};
    not used for \code{confint} and \code{plot}}
}
\details{
  The \code{AnotA} function uses the \code{glm} and \code{fisher.test}
  functions of the \code{stats} package. Note that all arguments have
  to be positive integers.
}
\value{
  For \code{AnotA} an object of class \code{anota} (which has a print
  method). This is a list with elements
  \item{coefficients}{ named vector of coefficients (d-prime)}
  \item{res.glm}{the glm-object from the fitting process}
  \item{vcov}{variance-covariance matrix of the coefficients}
  \item{se}{named vector with standard error of the coefficients
    (standard error of d-prime}
  \item{data}{a named vector with the data supplied to the function}
  \item{p.value}{one-sided p-value from Fisher's exact test
    (\code{fisher.test})}
  \item{test}{a string with the name of the test (\code{A-Not A}) for
    the print method}
  \item{call}{the matched call}

  For \code{plot} a figure of the distributions of sensory intensity is
  produced, and for \code{confint} a 2-by-2 matrix of confidence
  intervals is returned.
}
\references{Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian
  models for sensory discrimination tests as generalized linear models.
  Food Quality and Preference, 21, pp. 330-338.}
\author{Rune Haubo B Christensen and Per Bruun Brockhoff}

\seealso{\code{\link{print.discrim}}, \code{\link{discrim}},
  \code{\link{discrimPwr}}, \code{\link{discrimSim}},
  \code{\link{discrimSS}}, \code{\link{findcr}}}

\examples{
# data: 10 of the A-samples were judged to be A
#       20 A-samples in total
#       3 of the not-A samples were judged to be A
#       20 not-A-samples in total

AnotA(10, 20, 3, 20)
(m1 <- AnotA(10, 20, 3, 20))

## plot distributions of sensory intensity:
plot(m1)

## likelihood based confidence intervals:
confint(m1)

\dontshow{
  ## Testing
  x <- as.vector(c(coef(m1), m1$se, m1$p.value))
  ## y := dput(x)
  y <- c(1.03643338948988, 0.442500608613267, 0.0203711975124656)
  ## z := dput(as.vector(confint(m1)))
  z <- c(-0.550522667455557, 0.190017579324113, 0.550522667455557,
    1.93313337684111)
  stopifnot(
    isTRUE(all.equal(x, y)),
    isTRUE(all.equal(as.vector(confint(m1)), z))
  )
}
## Extended example plotting the profile likelihood
xt <- cbind(c(3, 10), c(20 - 3, 20 - 10))
lev <- gl(2, 1)
summary(res <- glm(xt ~ lev,
                   family = binomial(link = probit)))
N <- 100
dev <- double(N)
level <- c(0.95, 0.99)
delta <- seq(1e-4, 5, length = N)
for(i in 1:N)
  dev[i] <- glm(xt ~ 1 + offset(c(0, delta[i])),
                family = binomial(probit))$deviance
plot(delta, exp(-dev/2), type = "l",
     xlab = expression(delta),
     ylab = "Normalized Profile Likelihood")
## Add Normal approximation:
lines(delta, exp(-(delta - coef(res)[2])^2 /
                 (2 * vcov(res)[2,2])), lty = 2)
## Add confidence limits:
lim <- sapply(level, function(x)
              exp(-qchisq(x, df=1)/2) )
abline(h = lim, col = "grey")

}
\keyword{models}% at least one, from doc/KEYWORDS
