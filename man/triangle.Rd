\name{triangle}
\alias{triangle}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Create triangle binomial family}
\description{
  Creates af copy of the binomial family with the inverse link function
  changed to equal
  the triangle psychometric function and correspondingly changed link
  function and derivative of the
  inverse link function.}
\usage{
triangle()
}
\value{
  A binomial family object for models. Among other things it inludes the
  psychometric function as
  \code{linkinv} and the inverse psychometric function (for direct
  dprime computation) as
  \code{linkfun}.
}
\references{Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian
  models for sensory discrimination tests as generalized linear models.
  Food Quality and Preference, 21, pp. 330-338.}
\author{Rune Haubo B Christensen and Per Bruun Brockhoff}
\note{
  Several functions in this package makes use of the function, but it
  may also be used on its own---see the example below.
}

\seealso{\code{\link{duotrio}}, \code{\link{twoAFC}},
  \code{\link{tetrad}}, \code{\link{threeAFC}}, \code{\link{discrim}},
  \code{\link{discrimPwr}}, \code{\link{discrimSim}},
  \code{\link{AnotA}}, \code{\link{discrimSS}},
  \code{\link{samediff}}, \code{\link{findcr}}}

\examples{

## Estimating d-prime using glm for a Triangle test:
xt <- matrix(c(10, 5), ncol = 2) ## data: 10 correct answers, 5 incorrect
res <- glm(xt ~ 1, family = triangle)
summary(res)
## Equivalent to (Estimate and Std. Error):
discrim(10, 15, method="triangle")

\dontshow{
## Test agreement of glm and discrim results:
x <- coef(summary(res))[, 1:2]
y <- coef(discrim(10, 15, method="triangle"))["d-prime", 1:2]
stopifnot(
  isTRUE(all.equal(as.vector(x), as.vector(y), tol=1e-5)),
  isTRUE(all.equal(res$deviance, 0))
)
}
## Extended example plotting the profile likelihood
## data: 10 correct answers, 9 incorrect
xt <- matrix(c(10, 9), ncol = 2)
summary(res <- glm(xt ~ 1, family = triangle))
N <- 100
dev <- double(N)
delta <- seq(1e-4, 3, length = N)
for(i in 1:N)
  dev[i] <- glm(xt ~ -1 + offset(delta[i]),
                family = triangle)$deviance
plot(delta, exp(-dev/2), type = "l",
     xlab = expression(delta),
     ylab = "Normalized Profile Likelihood")
## Add Normal approximation:
lines(delta, exp(-(delta - coef(res))^2 /
                 (2 * vcov(res))), lty = 2)
## Add confidence limits:
level <- c(0.95, 0.99)
lim <- sapply(level, function(x) exp(-qchisq(x, df=1)/2) )
abline(h = lim, col = "grey")

}
\keyword{models}% at least one, from doc/KEYWORDS

