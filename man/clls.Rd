\name{clls-deprecated}
\alias{clls-deprecated}
\alias{clls}

\title{
  Cumulative Link Location-Scale Models
}
\description{
  IMPORTANT: This function and its methods are no longer supported. The
  user is adviced to use clm() from package ordinal instead.

  Fits a cumulative link location-scale model to an ordered response
  variable. When the scale part is left unspecified, the model reduces
  to a cumulative link model assuming a constant scale. With the default
  logistic link function, the model reduces to the famous \emph{Proportional
  Odds Model}. With the probit link and a single two-level factor in both
  location and scale parts, the model is known as the \emph{Binormal}
  model in the Signal Detection Theory and the Psychometric
  literature.
}
\usage{
  clls(location, scale, data, weights, start, ..., subset,
           na.action, contrasts = NULL, Hess = FALSE, model = TRUE,
           method = c("logistic", "probit", "cloglog", "cauchit"))
}
\arguments{
\item{location}{
  a formula expression as for regression models, of the form
  \code{response ~ predictors}. The response should be a factor
  (preferably an ordered factor), which will be interpreted as an
  ordinal response, with levels ordered as in the factor.
  The model must have an intercept: attempts to remove one will
  lead to a warning and be ignored. An offset may be used.  See the
  documentation of \code{\link{formula}} for other details.
}
\item{scale}{
  a optional formula expression as for the location part, of the form
  \code{ ~ predictors}, ie. with an empty left hand side.
  If left unspecified, the model assumes a constant scale and reduces
  to the cumulative link model.
  An offset may be used.  See the
  documentation of \code{\link{formula}} for other details.
}
\item{data}{
  an optional data frame in which to interpret the variables occurring
  in \code{formula}.
}
\item{weights}{
  optional case weights in fitting.  Default to 1.
}
\item{start}{
  initial values for the parameters.  This is in the format
  \code{c(beta, theta, sigma)}: see the Values section.
}
\item{\dots}{
  additional arguments to be passed to \code{\link{optim}}, most often a
  \code{control} argument.
}
\item{subset}{
  expression saying which subset of the rows of the data should  be used
  in the fit.  All observations are included by default.
}
\item{na.action}{
  a function to filter missing data.
}
\item{contrasts}{
  a list of contrasts to be used for some or all of
  the factors appearing as variables in the model formula.
}
\item{Hess}{
  logical for whether the Hessian (the observed information matrix)
  should be returned.  Use this if you intend to call \code{summary} or
  \code{vcov} on the fit.
}
\item{model}{
  logical for whether the model matrix should be returned.
}
\item{method}{
  logistic or probit or complementary log-log or cauchit (corresponding
  to a Cauchy latent variable).
}
}
\details{
  The implementation is highly inspired by \code{\link[MASS]{polr}} in
  package MASS and should give compatible results, if \code{scale} is
  left unspecified.

  Note that standard errors are appropriate for \code{tau} =
  log \code{sigma} and not for \code{sigma}, because the profile
  likelihood is usually more symmetric for \code{tau} than for
  \code{sigma}. Therefore \code{vcov} will give the
  variance-covariance matrix of the parameters with \code{tau} rather
  than \code{sigma} and \code{summary.clls} will report standard errors
  for log \code{sigma}. Notice also that a relevant test for
  \code{sigma} is \eqn{H_0: sigma = 1}, so the relevant test for log
  \code{sigma} is \eqn{H_0: log(sigma) = 0}. This is reflected in the z
  value for \code{sigma} returned by \code{summary.clls}.

  There are methods for the standard model-fitting functions, including
  \code{\link{summary}}, \code{\link{vcov}},
  \code{\link{anova}}, and an
  \code{extractAIC} method.
}
\value{
  A object of class \code{"clls"}. This has components

  \item{coefficients}{the coefficients of the location
    (\code{beta}), the intercepts (\code{theta}) and the scale
    (\code{sigma}).}
  \item{beta}{the parameter estimates of the location part.}
  \item{theta}{the intercepts/thresholds for the class boundaries.}
  \item{sigma}{the parameter estimates of the scale part.}
  \item{tau}{parameter estimates of the scale part on the log scale;
    ie. \code{tau} = log \code{sigma}.}
  \item{deviance}{the residual deviance.}
  \item{fitted.values}{a matrix, with a column for each level of the
    response with the fitted probabilities.}
  \item{fitted.case}{a vector of same length as \code{response}, with
    the fitted probabilities on a case-by-case basis.}
  \item{lev}{the names of the response levels.}
  \item{terms.location}{a \code{terms} structure describing the location
    part.}
  \item{terms.scale}{a \code{terms} structure describing the scale
    part.}
  \item{df.residual}{the number of residual degrees of freedoms,
    calculated using the weights.}
  \item{edf}{the (effective) number of degrees of freedom used by the
    model}
  \item{n, nobs}{the (effective) number of observations, calculated
    using the weights.}
  \item{call}{the matched call.}
  \item{method}{the matched method used.}
  \item{convergence}{the convergence code returned by \code{optim}.}
  \item{niter}{the number of function and gradient evaluations used by
    \code{optim}.}
  \item{Hessian}{if \code{Hess} is true, the observed Fisher information
    matrix.}
  \item{location}{if \code{model} is true, the \code{model.frame} for
    the location part.}
  \item{scale}{if \code{model} is true, the \code{model.frame} for
    the scale part.}
}
\references{
  Agresti, A. (2002) \emph{Categorical Data.} Second edition.  Wiley.

  Christensen, R. H. B., Brockhoff, P. B. and Cleaver, G. (2008)
  Estimation and Inference in the A-Not A test with Sureness. Manuscript
  for \emph{Food Quality and Preference}.

  Venables, W. N. and Ripley, B. D. (2002)
  \emph{Modern Applied Statistics with S.} Fourth edition.  Springer.
}
\seealso{
  \code{\link[MASS]{polr}}, \code{\link{optim}}, \code{\link{glm}},
  \code{\link[nnet]{multinom}}.
}
\examples{
\dontrun{
  options(contrasts = c("contr.treatment", "contr.poly"))
  ## Extend example from polr in package MASS:
  ## Fit model from polr example:
  data(housing, package = "MASS")
  fm1 <- clls(Sat ~ Infl + Type + Cont, weights = Freq, data = housing)
  fm1
  summary(fm1)
  ## With probit link:
  summary(update(fm1, method = "probit"))

  ## Allow scale to depend on Cont-variable
  summary(fm2 <- update(fm1, scale =~ Cont))
  anova(fm1, fm2)
  ## which seems to improve the fit
}

}
\keyword{models}
