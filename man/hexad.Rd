\name{hexad}
\alias{hexad}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Create hexad binomial family}
\description{
  Creates af binomial family object with the inverse link function
  equal to the psychometric function for the unspecified Hexad test.
}
\usage{

hexad()

}
\value{
  A binomial family object for models. Among other things it inludes the
  psychometric function as
  \code{linkinv} and the inverse psychometric function (for direct
  d-prime computation) as
  \code{linkfun}.
}
\references{
  Eberhardt, K., Aubry, V., & Robinson, K. (2008). 
  A thurstonian model for the unspecified hexad test.
  In 2008 Sensometrics Meeting ’Discover a New World of Data’ (E-5). 
  Kraft Foods.

%%   Brockhoff, P.B. and Christensen, R.H.B. (2010). Thurstonian
%%   models for sensory discrimination tests as generalized linear models.
%%   Food Quality and Preference, 21, pp. 330-338.
%%
}
\author{Karolina Stachlewska}
\note{
  Several functions in this package makes use of functions in the hexad
  family object, but it may also be used on its own---see the example
  below.
}

\seealso{\code{\link{duotrio}}, \code{\link{triangle}}, \code{\link{twoAFC}},
  \code{\link{threeAFC}}, \code{\link{tetrad}}, \code{\link{twofive}},
   \code{\link{twofiveF}}, \code{\link{discrim}},
  \code{\link{discrimPwr}}, \code{\link{discrimSim}},
  \code{\link{AnotA}}, \code{\link{discrimSS}},
  \code{\link{samediff}}, \code{\link{findcr}}}

\examples{

## Estimating d-prime using glm for an unspecified Hexad test:
xt <- matrix(c(10, 5), ncol = 2) ## data: 10 correct answers, 5 incorrect
res <- glm(xt ~ 1, family = hexad)
summary(res)
## Equivalent to (Estimate and Std. Error):
discrim(10, 15, method="hexad")

\dontshow{
## Test agreement of glm and discrim results:
x <- coef(summary(res))[, 1:2]
y <- coef(discrim(10, 15, method="hexad"))["d-prime", 1:2]
stopifnot(
  isTRUE(all.equal(as.vector(x), as.vector(y), tol=1e-5)),
  isTRUE(all.equal(res$deviance, 0))
)
}
## Extended example plotting the profile likelihood
## data: 10 correct answers, 9 incorrect
xt <- matrix(c(10, 9), ncol = 2)
summary(res <- glm(xt ~ 1, family = hexad))
N <- 100
dev <- double(N)
delta <- seq(1e-4, 3, length = N)
for(i in 1:N)
  dev[i] <- glm(xt ~ -1 + offset(delta[i]),
                family = hexad)$deviance
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

