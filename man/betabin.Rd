\name{betabin}
\alias{betabin}
\alias{summary.betabin}
\title{Beta-binomial and chance-corrected beta-binomial models for
  over-dispersed binomial data}
\description{
  Fits the beta-binomial model and the chance-corrected beta-binomial
  model to (over-dispersed) binomial data.
}
\usage{
betabin(data, start = c(.5,.5),
        method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
          "triangle"),
        vcov = TRUE, corrected = TRUE, gradTol = 1e-4, ...)

\method{summary}{betabin}(object, level = 0.95, ...)

}
\arguments{
  \item{data}{matrix or data.frame with two columns; first column
    contains the number of success and the second the total number of
    cases. The number of rows should correspond to the number of
    observations.}
  \item{start}{starting values to be used in the optimization}
  \item{vcov}{logical, should the variance-covariance matrix of the
    parameters be computed?}
  \item{method}{the sensory discrimination protocol for which d-prime
    and its standard error should be computed}
  \item{corrected}{should the chance corrected or the standard beta
    binomial model be estimated?}
  \item{gradTol}{a warning is issued if max|gradient| < gradTol, where
    'gradient' is the gradient at the values at which the optimizer
    terminates. This is not used as a termination or convergence
    criterion during model fitting.}
  \item{object}{an object of class "betabin", i.e. the result of
    \code{betabin()}.}
  \item{level}{the confidence level of the confidence intervals computed
    by the summary method}
  \item{\dots}{\code{betabin}: The only recognized (hidden) argument is
    \code{doFit} (boolean) which by default is \code{TRUE}. When
    \code{FALSE} \code{betabin} returns an environment which facilitates
    examination of the likelihood surface via the (hidden) functions
    \code{sensR:::getParBB} and \code{sensR:::setParBB}.
    Not used in \code{summary.betabin}.}
}
\details{
The beta-binomial models are parameterized in terms of mu and gamma,
where mu corresponds to a probability parameter and gamma measures
over-dispersion. Both parameters are restricted to the interval (0, 1).
The parameters of the standard (i.e. corrected = FALSE) beta-binomial
model refers to the mean (i.e. probability) and dispersion on the scale
of the observations, i.e. on the scale where we talk of a probability of
a correct answer (Pc). The parameters of the chance corrected (i.e. corrected
= TRUE) beta-binomial model refers to the mean and dispersion on the
scale of the "probability of discrimination" (Pd).
The mean parameter (mu) is therefore restricted to the
interval from zero to one in both models, but they have different
interpretations.

The summary method use the estimate of mu to infer the parameters of the
sensory experiment; Pc, Pd and d-prime. These are restricted to their
allowed ranges, e.g. Pc is always as least as large as the guessing
probability.

Confidens intervals are computed as Wald (normal-based) intervals on the
mu-scale and the confidence limits are subsequently transformed to the
Pc, Pd and d-prime scales. Confidence limits are restricted to the
allowed ranges of the parameters, for example no confidence limits will
be less than zero.

Standard errors, and therefore also confidence intervals, are only
available if the parameters are not at the boundary of their allowed
range (parameter space). If parameters are close to the boundaries of
their allowed range, standard errors, and also confidence intervals, may
be misleading. The likelihood ratio tests are more accurate. More
accurate confidence intervals such as profile likelihood intervals may
be implemented in the future.

The summary method provides a likelihood ratio test of over-dispersion
on one degree of freedom and a likelihood ratio test of association
(i.e. where the null hypothesis is "no difference" and the alternative
hypothesis is "any difference") on two degrees of
freedom (chi-square tests). Since the gamma parameter is tested on the
boundary of the
parameter space, the correct degree of freedom for the first test is
probably 1/2 rather than one, or somewhere in between, and the latter
test is probably also on less than two degrees of freedom. Research is
needed to determine the appropriate no. degrees of freedom to use in
each case. The choices used here are believed to be conservative, so the
stated p-values are probably a little too large.

The log-likelihood of the standard beta-binomial model is
\deqn{\ell(\alpha, \beta; x, n) =
  \sum_{j=1}^N \left\{
  \log {n_j \choose x_j}
  - \log Beta(\alpha, \beta)
  + \log  Beta(\alpha + x_j, \beta - x_j + n_j) \right\}
  }

and the log-likelihood of the chance corrected beta-binomial model is
\deqn{\ell(\alpha, \beta; x, n) =
  \sum_{j=1}^N \left\{ C
  + \log \left[ \sum_{i=0}^{x_j} {{x_j} \choose i}
  (1-p_g)^{n_j-x_j+i} p_g^{x_j-i}
  Beta(\alpha + i, n_j - x_j + \beta) \right] \right\}
}
where
\deqn{C =
  \log {n_j \choose x_j}
  - \log Beta(\alpha, \beta)
}

and where \eqn{\mu = \alpha/(\alpha + \beta)},
\eqn{\gamma = 1/(\alpha + \beta + 1)}, \eqn{Beta} is the Beta
function, cf. \code{\link{beta}},
\eqn{N} is the number of independent binomial observations, i.e.~the
number of rows in \code{data}, and \eqn{p_g} is the guessing
probability, \code{pGuess}.

The variance-covariance matrix (and standard errors) is based on the
inverted Hessian at the optimum. The Hessian is obtained with the
\code{hessian} function from the numDeriv package.

The gradient at the optimum is evaluated with \code{gradient} from the
numDeriv package.

The bounded optimization is performed with the "L-BFGS-B" optimizer in
\code{\link{optim}}.

The following additional methods are implemented objects of class
\code{betabin}: \code{print}, \code{vcov} and \code{logLik}.
}
\value{
  An object of class \code{betabin} with elements
  \item{coefficients}{named vector of coefficients}
  \item{vcov}{variance-covariance matrix of the parameter estimates if
    \code{vcov = TRUE}}
  \item{data}{the data supplied to the function}
  \item{call}{the matched call}
  \item{logLik}{the value of the log-likelihood at the MLEs}
  \item{method}{the method used for the fit}
  \item{convergence}{0 indicates convergence. For other error messages,
    see \code{\link{optim}}.}
  \item{message}{possible error message - see \code{\link{optim}} for
    details}
  \item{counts}{the number of iterations used in the optimization - see
    \code{\link{optim}} for details}
  \item{corrected}{is the chance corrected model estimated?}
  \item{logLikNull}{log-likelihood of the binomial model with
    prop = pGuess}
  \item{logLikMu}{log-likelihood of a binomial model with
    prop = sum(x)/sum(n)}
}
\references{Brockhoff, P.B. (2003). The statistical power of
  replications in difference tests.
  Food Quality and Preference, 14, pp. 405--417.
}

\author{Rune Haubo B Christensen}

\seealso{ \code{\link{triangle}}, \code{\link{twoAFC}},
  \code{\link{threeAFC}}, \code{\link{duotrio}}, \code{\link{tetrad}}
}

\examples{
## Create data:
x <- c(3,2,6,8,3,4,6,0,9,9,0,2,1,2,8,9,5,7)
n <- c(10,9,8,9,8,6,9,10,10,10,9,9,10,10,10,10,9,10)
dat <- data.frame(x, n)

## Chance corrected beta-binomial model:
(bb0 <- betabin(dat, method = "duotrio"))
summary(bb0)
## Un-corrected beta-binomial model:
(bb <- betabin(dat, corrected = FALSE, method = "duotrio"))
summary(bb)
vcov(bb)
logLik(bb)
AIC(bb)
coef(bb)
\dontshow{
## Testing:
## Chance corrected model:
x <- as.vector(coef(summary(bb0)))
## x2 := dput(x)
x2 <- c(0.176050917183022, 0.504272625540766, 0.588025458591511, 0.176050917183022,
1.03723538712345, 0.0827416835966564, 0.166754897796229, 0.0413708417983282,
0.0827416835966564, 0.274652313643027, 0.0138801973133669, 0.177439031614499,
0.506940098656683, 0.0138801973133669, 0.276003094947466, 0.338221637052677,
0.831106219467033, 0.669110818526339, 0.338221637052677,
1.53322252140796)

stopifnot(isTRUE(all.equal(x, x2)))

## Un-corrected model:
x <- as.vector(coef(summary(bb)))
## x2 := dput(x)
x2 <- c(0.493812345858779, 0.31442578341548, 0.5, 0, 0, 0.0723816596143461,
0.0861846136283124, NA, NA, NA, 0.351946899873424, 0.145507044682488,
0.5, 0, 0, 0.635677791844135, 0.483344522148472, 0.635677791844135,
0.27135558368827, 1.33527484290736)
stopifnot(isTRUE(all.equal(x, x2)))
}
}
\keyword{models}% at least one, from doc/KEYWORDS
