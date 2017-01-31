\name{profile.samediff}
\alias{profile.samediff}
\alias{plot.profile.samediff}
\alias{confint.samediff}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Profile likelihood methods for samediff objects.}
\description{
  Computes the (normalized or relative) profile likelihood for the
  parameters of a same-different test, plots the normalized profile
  likelihood and computes profile likelihood confidence intervals.
}
\usage{
\method{profile}{samediff}(fitted, which = 1:2, max = 2, numpts = 100,
           max.delta = 10, max.tau = 10, ...)
\method{plot}{profile.samediff}(x, which = 1:nc, level = c(0.99, 0.95),
           fig = TRUE, ...)
\method{confint}{samediff}(object, parm = c("tau", "delta"), level = 0.95, max = c(10, 10)
           , ...) 
}
\arguments{
  \item{fitted}{a \code{samediff} object}
  \item{x}{a \code{profile.samediff} object}
  \item{object}{a \code{samediff} object}
  \item{which}{numeric: which parameters to profile or plot; either "1" or "2"
    or "1:2" to mean "tau", "delta" or both respectively.}
  \item{parm}{the parameter(s) to compute the confidence interval for}
  \item{max}{for \code{profile}: control parameter to specify how many units beyond the MLE,
    the profiling should proceed. For \code{confint}: control parameter,
    that can control the convergence for especially very large \code{delta}}
  \item{numpts}{control parameter: At how many points should the profile
    likelihood be evaluated?}
  \item{max.delta}{control parameter: The maximum point at which to
    evaluate the profile likelihood for delta}
  \item{max.tau}{same as \code{max.delta} for "tau".}
  \item{level}{for \code{plot}: At which levels to include horizontal lines to indicate
    confidence levels in plots of the normalized profile
    likelihoods. For \code{confint}: at which level to compute the
    confidence interval.}
  \item{fig}{logical: Should the normalized profile likelihoods be plotted?}
  \item{\dots}{not currently used.}
}
%% \details{
%%   Note that the variance-covariance matrix does not always exist in
%%   contrast to the profile likelihood. \code{profile = FALSE} may
%%   therefore cause confidence intervals etc. to be \code{NA}.
%% }
\value{
  For \code{profile}:
  An object of class \code{"profile.samediff", "data.frame"}---a
  \code{data.frame} with two columns for each parameter profiled giving
  the value of the parameter and the corresponding value of the profile
  likelihood.

  For \code{plot}:
  An object of class \code{"nProfile.samediff", "data.frame"}---the
  \code{data.frame} from the \code{profile}-object with extra
  columns corresponding to the \code{which} parameter containing the
  normalized profile liklelihood.

  For \code{confint}:
  A 2x2 matrix with columns named \code{"lower", "upper"} giving the
  lower and upper (1 - \code{alpha})\% confidence interval for the
  parameters named in the rows.
}
\author{Rune Haubo B Christensen}

\seealso{
  \code{\link{summary.samediff}}
}

\examples{
# data: 8 of the same samples were judged to be same
#       5 of the same samples were judged to be different
#       4 of the different samples were judged to be same
#       9 of the different samples were judged to be different

sadi <- samediff(8, 5, 4, 9)
confint(sadi)
plot(profile(sadi))

}
\keyword{models}% at least one, from doc/KEYWORDS
