\name{discrimR}
\alias{discrimR}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Replicated Thurstonian Model for discrimination analysis}
\description{
  The model is a synthesis of a mixture and a mixed effect model. The
  random effect distribution for the cluster term (often individuals) is
  a point mass for delta = 0 and a continuous distribution for delta >
  0.

  The function fits the model and computes d-prime for an average
  subject, 2) the variance among subjects, 3) the "posterior" probability
  of a subject being a discriminator (with delta > 0), 4) the
  "posterior" expectation on the random effect (ie. the subject-specific
  delta) and 5) the probability that a randomly chosen individual is a
  discriminator (ie. the probability mass at delta = 0 in the random
  effects distribution)

  Warning: This function is preliminary; see the details for further
  information.
}
\usage{
discrimR(formula, data, weights, cluster, start, subset, na.action,
           contrasts = NULL, hess = FALSE, ranef = FALSE, zi = FALSE,
           method = c("duotrio", "probit", "threeAFC", "triangle",
             "twoAFC"), ...)
}
\arguments{
  \item{formula}{A formula where the lhs is the binomial response. An
    indicator vector or a matrix with two column; successes and failures
    like in a call to \code{\link{glm}} with a binomial family. The rhs
    should be \code{1}; no other predictors are currently allowed, but
    extending this is ongoing work.}
  \item{data}{The \code{data.frame} in which to look for variables.}
  \item{weights}{Possible weights}
  \item{cluster}{The clustering variable; should be a factor.}
  \item{start}{Optional starting values; recommended in the current
    implementation} 
  \item{subset}{...}
  \item{na.action}{...}
  \item{contrasts}{...}
  \item{hess}{Should the hessian of the parameters be computed?}
  \item{ranef}{Should the random effect estimates be computed?}
  \item{zi}{Should the posterior probabilities of a subject being a
    discriminator be computed? }
  \item{method}{Should correspond to the actual test applied.}
  \item{\dots}{Additional arguments to
    \code{\link{optim}}. \code{control=list(trace=TRUE, REPORT=1)} is
    recommended, so the reduction in deviance and convergence can be
    followed.}
}
\details{
  This function is preliminary and improving it is ongoing work. The
  computational methods are expected to change completely. This will
  hopefully facilitate methods for more general rhs-formulae with
  additional predictors.

  Currently no methods or extractor functions have been written, so the
  user will have to select the relevant elements from the fitted object
  (see below). Implementation of methods and extractor functions will
  occur in due course. 
}
\value{
  A list with the following elements:
  \item{fpar}{The fixed effect parameter, ie. delta (for an average
    individual)} 
  \item{rpar}{A vector with two elements: The first element is the
    variance component (standard deviation) on the log-scale, where
    optimization is performed. The second element is the variance
    component (standard deviation) on the original scale.}
  \item{deviance}{Deviance for the model}
  \item{se}{standard errors for 1) the fixed effect parameter and 2) the
    variance component on the log-scale}
  \item{convergence}{Convergence message from \code{\link{optim}}}
  \item{lli}{Log-likelihood contributions from each of the observations.}
  \item{ranef}{The random effect estimates for the levels of the
    clustering factor (often individual)}
  \item{zi}{posterior probabilities of a subject being a
    discriminator}
  \item{p}{The probability that a randomly chosen individual is a
    discriminator (ie. the probability mass for delta > 0 in the random
    effects distribution)}
  \item{fitted}{Fitted values}
  \item{Y}{The scaled response vector on which optimization is performed.}
  \item{call}{the matched call}
}
\author{Rune Haubo B Christensen}

\seealso{ \code{\link{triangle}}, \code{\link{twoAFC}},
  \code{\link{threeAFC}}, \code{\link{duotrio}},
  \code{\link{discrimPwr}}, \code{\link{discrimSim}},
  \code{\link{discrimSS}}, \code{\link{samediff}},
  \code{\link{AnotA}}, \code{\link{findcr}}
}

\examples{
\dontrun{
freq <- c(10,8,10,9,8,9,9,1,10,10,8,2,6,7,6,7,6,4,5,5,3,3,9,9,5,5,8,8,9,9)
tmp <- data.frame(id = factor(1:30), n = rep(10, 30), freq = freq)
head(tmp)
str(tmp)

fm <- discrimR(cbind(freq, n - freq) ~ 1, tmp, cluster = id,
                    start = c(.5, .5), method = "twoAFC",
                    ranef = TRUE, zi = TRUE, hess = TRUE, 
                    control=list(trace=TRUE, REPORT=1))

names(fm)
fm[1:4]
}
}
\keyword{models}% at least one, from doc/KEYWORDS
