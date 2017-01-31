\name{discrim}
\alias{discrim}
\alias{print.discrim}
% \alias{discrimr}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Sensory discrimination analysis}
\description{
  Computes the probability of a correct answer (Pc), the probability of
  discrimination (Pd) and d-prime, their standard errors, confidence
  intervals and a p-value of a difference or similarity test for one of
  the four common discrimination protocols.
}
\usage{

discrim(correct, total, d.prime0, pd0, conf.level = 0.95,
           method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
             "triangle"),
           statistic = c("exact", "likelihood", "score", "Wald"),
           test = c("difference", "similarity"), ...)

\method{print}{discrim}(x, digits = max(3, getOption("digits")-3), ...)

%% discrimr(formula, data, weights, start, subset, na.action,
%%          contrasts = NULL, method = c("duotrio", "probit", "threeAFC",
%%            "triangle", "twoAFC", "logit"), Hess = TRUE, ...)
}
\arguments{
  \item{correct}{the number of correct answers; non-negativescalar
    integer}
  \item{total}{the total number of answers (the sample size); positive
    scalar integer}
  \item{d.prime0}{The value of d-prime under the
    null hypothesis; numerical non-zero scalar}
  \item{pd0}{the probability of discrimination under the
    null hypothesis; numerical scalar between zero and one}
%%%   \item{formula}{an object of class "formula" (or one that can be coerced
%%%     to that class): a symbolic description of the model to be fitted.}
%%%   \item{data}{an optional data frame, list or environment (or object coercible
%%%     by as.data.frame to a data frame) containing the variables in the
%%%     model. If not found in data, the variables are taken from
%%%     environment(formula)}
%%%   \item{start}{starting values to be used in the optimization}
%%%   \item{subset}{an optional vector specifying a subset of observations to be
%%%     used in the fitting process.}
%%%   \item{weights}{an optional vector of weights to be used in the fitting
%%%     process. Should be NULL or a numeric vector.}
%%%   \item{na.action}{a function which indicates what should happen when the
%%%     data contain NAs. The default is set by the na.action setting of
%%%     options, and is na.fail if that is unset. The "factory-fresh" default
%%%     is na.omit. Another possible value is NULL, no action. Value na.exclude
%%%     can be useful.}
%%%   \item{contrasts}{an optional list of specifying the contrasts to use.}
%%%   \item{hess}{logical, should the hessian be computed? (requires for
%%%     standard errors etc. to be computed by summary.discrimr)}
  \item{conf.level}{the confidence level for the confidence intervals}
  \item{method}{the discrimination protocol. Five allowed values:
    "twoAFC", "threeAFC", "duotrio", "tetrad", "triangle"}
  \item{test}{the type of test}
  \item{statistic}{the statistic to be used for hypothesis testing and
    confidence intervals}
  \item{x}{an object of class \code{"discrim"}}
  \item{digits}{number of digits in resulting table of results}
  \item{\dots}{not currently used}
}
\details{
  The degree of product difference/discrimination under the null
  hypothesis can be specified on \emph{either} the d-prime scale or on
  the pd (proportion of discriminators) scale. This is done by using
  either the \code{d.prime0} \emph{or} the \code{pd0} arguments.
  If unspecified, they default to zero and the
  conventional difference test of "no difference" is obtained.

  For a similarity test either \code{d.prime0} \emph{or} \code{pd0} have
  to be specified \emph{and} and a non-zero, positive value should to be
  given. Here, \code{d.prime0} or \code{pd0} define the \code{limit of
  similarity} or \code{equivalence}.

  The probability under the null hypothesis is
  given by \code{pd0 + pg * (1 - pd0)} where \code{pg} is the guessing
  probability which is defined by the discrimination protocol given in
  the \code{method} argument.

  All estimates are restricted to their allowed ranges, e.g. Pc is
  always as least as large as the guessing probability. Similarly
  confidence limits are also restricted to the allowed range of the
  parameters.

  Standard errors are not defined when the parameter estimates are at
  the boundary of their allowed range, so these will be reported as
  \code{NA} in such cases.

  The \code{"Wald"} statistic is *NOT* recommended for practical
  use---it is included here for completeness and to allow comparisons.

  For \code{statistic = "score"}, the confidence interval is computed
  from Wilson's score interval, and the p-value for the hypothesis
  test is based on Pearson's chi-square test,
  cf. \code{\link{prop.test}}.

}
\value{
  An object of class \code{discrim} with elements
  \item{coefficients}{matrix of estimates, standard errors and
    confidence intervals}
  %% \item{res.glm}{the glm-object from the fitting process}
  %% \item{vcov}{variance-covariance matrix of the coefficients}
  %% \item{se}{named vector with standard error of the coefficients
  %%   (standard error of d-prime)}
  \item{data}{a named vector with the data supplied to the function}
  \item{p.value}{the p-value of the hypothesis test}
  \item{call}{the matched call}
  \item{test}{the type of test}
  \item{method}{the discrimination protocol}
  \item{statistic}{the statistic used for confidence intervals and
    p-value}
  \item{pd0}{the probability of discrimination under the
    null hypothesis}
  \item{alt.scale}{the scale for the alternative hypothesis,
    e.g.~\code{"d.prime"} or \code{"pd"}}
  \item{conf.level}{the confidence level}
  \item{stat.value}{for \code{statistic != "exact"} the value of the
    test statistic used to calculate the p-value}
  \item{df}{for \code{statistic == "score"} the number of degrees of
    freedom used for the Pearson chi-square test to calculate the
    p-value}
  \item{profile}{for \code{statistic == "likelihood"} the profile
    likelihood on the scale of Pc}
}
\references{Brockhoff, P.B. and Christensen, R.H.B (2010). Thurstonian
  models for sensory discrimination tests as generalized linear models.
  Food Quality and Preference, 21, pp. 330-338.}
\author{Rune Haubo B Christensen and Per Bruun Brockhoff}

\seealso{ \code{\link{triangle}}, \code{\link{twoAFC}},
  \code{\link{threeAFC}}, \code{\link{duotrio}},
  \code{\link{tetrad}},
  \code{\link{discrimPwr}}, \code{\link{discrimSim}},
  \code{\link{discrimSS}}, \code{\link{samediff}},
  \code{\link{AnotA}}, \code{\link{findcr}},
  \code{\link[=profile.discrim]{profile}},
  \code{\link[=profile.discrim]{plot.profile}}
  \code{\link[=profile.discrim]{confint}}
}

\examples{
## Running the simple discrimination (difference) tests:
discrim(10, 15, method = "twoAFC")
discrim(10, 15, method = "threeAFC", statistic = "likelihood")
discrim(10, 15, method = "tetrad", statistic = "likelihood")
discrim(10, 15, method = "duotrio", conf.level = 0.90)
discrim(10, 15, method = "triangle", statistic = "score")

## plot the distributions of sensory intensity:
m1 <- discrim(10, 15, method = "twoAFC")
plot(m1)

## A similarity test where less than chance successes are obtained:
discrim(22, 75, method = "triangle", d.prime0 = 1, test = "similarity")
\dontshow{
## Testing stability of results:
fm1 <- discrim(10, 15, method = "twoAFC")
fm2 <- discrim(10, 15, method = "threeAFC", statistic = "likelihood")
fm3 <- discrim(10, 15, method = "tetrad", statistic = "likelihood")
fm4 <- discrim(10, 15, method = "duotrio", conf.level = 0.90)
fm5 <- discrim(10, 15, method = "triangle", statistic = "score")
fm6 <- discrim(20, 75, method = "triangle", pd0 = .2, test = "similarity")

## str(fm1)
nm <- c("coefficients", "p.value")
mlist <- list(fm1, fm2, fm3, fm4, fm5, fm6)
x <- lapply(mlist, function(m) as.vector(unlist(m[nm])))
## xres := dput(x)
xres <- list(c(0.666666666666667, 0.333333333333333, 0.609140388347971,
0.121716123890037, 0.243432247780074, 0.473412265776244, 0.5,
0, 0, 0.881758896633119, 0.763517793266238, 1.67418186000377,
0.15087890625), c(0.666666666666667, 0.5, 1.11590231648908, 0.121716123890037,
0.182574185835055, 0.435915308196464, 0.415453723121119, 0.123180584681679,
0.280277592554894, 0.865219389059262, 0.797829083588893, 1.99667785632887,
0.00423459825376749), c(0.666666666666667, 0.5, 1.58778376949614,
0.121716123890037, 0.182574185835055, 0.419305544967608, 0.415453723121119,
0.123180584681679, 0.690727050515248, 0.865219389059262, 0.797829083588893,
2.41533835643047, 0.00423459825376749), c(0.666666666666667,
0.333333333333333, 1.51885152352487, 0.121716123890037, 0.243432247780074,
0.71586308378571, 0.5, 0, 0, 0.858336028357847, 0.716672056715693,
2.8242468406221, 0.15087890625), c(0.666666666666667, 0.5, 2.32136197974359,
0.121716123890037, 0.182574185835055, 0.651036422864387, 0.386865453268331,
0.0802981799024962, 0.782701393792843, 0.870121962915835, 0.805182944373752,
3.70692782369343, 0.00308494966027208), c(0.333333333333333,
0, 0, NA, NA, NA, 0.333333333333333, 0, 0, 0.381372720482923,
0.0720590807243847, 0.739487700025965, 0.000306144022107533))

for(i in 1:length(x)) stopifnot(isTRUE(all.equal(x[[1]], xres[[1]])))

}
}
\keyword{models}% at least one, from doc/KEYWORDS
