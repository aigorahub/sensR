\name{confint.twoAC}
\alias{confint.twoAC}
\alias{confint.profile.twoAC}
\alias{profile.twoAC}
\alias{plot.profile.twoAC}
\title{
  Confidence intervals and profile likelihoods for parameters in
  2AC models
}
\description{
  Computes confidence intervals from the profiled likelihood and the
  Wald approximation in the 2AC model, or plots the
  profile likelihood function for d.prime.
}
\usage{
\method{confint}{twoAC}(object, parm, level = 0.95,
        type = c("likelihood", "Wald"), \dots)

\method{confint}{profile.twoAC}(object, parm = "d.prime", level = 0.95, \dots)

\method{profile}{twoAC}(fitted, alpha = 1e-3, nSteps = 1e2, range, \dots)

\method{plot}{profile.twoAC}(x, level = c(0.95, 0.99), Log = FALSE,
      relative = TRUE, fig = TRUE, n = 1e3, ..., ylim = NULL)
}
\arguments{
  \item{object}{
    a fitted \code{\link{twoAC}} object or a \code{profile.twoAC} object.
  }
  \item{fitted}{
    a fitted \code{\link{twoAC}} object.
  }
  \item{x}{a \code{profile.twoAC} object.
  }
  \item{type}{
    the type of confidence interval required. \code{"profile"} is the
      most accurate.
  }
  \item{parm}{
    For \code{confint.profile.twoAC}:
    has to be \code{"d.prime"}.

    For \code{confint.twoAC}:
    for \code{type = "Wald"} a specification of which parameters the
    confidence interval is required for. Ignored for \code{type =
    "profile"}.
  }
  \item{level}{
    the confidence level required.
  }
  \item{alpha}{determines the range of profiling. By default the
    likelihood is profiled in the 99.9\% Wald confidence interval
    region.
  }
  \item{range}{
    if supplied, \code{d.prime} will be profiled between
    \code{min(range)} and \code{max(range)}. This over-rules the
    automatic range computation.
  }
  \item{nSteps}{the number of profile steps.
  }
  \item{Log}{should the profile likelihood be plotted on the log-scale?
  }
  \item{relative}{should the relative or the absolute likelihood be
    plotted?
  }
  \item{fig}{should the profile likelihood be plotted?
  }
  \item{n}{the no. points used in the spline interpolation of the
    profile likelihood.
  }
  \item{ylim}{overrules default y-limits on the plot of the profile
    likelihood.
  }
  \item{\dots}{
    not currently used.
  }
}
\value{
  \code{confint}:
  A matrix (or vector) with columns giving lower and upper confidence
  limits for each parameter. These will be labelled as (1-level)/2 and
  1 - (1-level)/2 in \% (by default 2.5\% and 97.5\%).
  Profile likelihood confindence intervals are only available for
  \code{d.prime} and not \code{tau}.

  \code{profile.twoAC}: a \code{data.frame} with the profile of
  \code{d.prime}.

  \code{plot.profile.twoAC} invisibly returns the spline approcimation to
  the profile.

}
\details{
  These \code{confint} methods call
  the appropriate profile method, then finds the
  confidence intervals by interpolation of the profile traces.
  If the profile object is already available, this should be used as the
  main argument rather than the fitted model object itself.

  In \code{plot.profile.twoAC}: at least one of \code{Log} and
  \code{relative} arguments have to be \code{TRUE}.

}
\references{
  Christensen R.H.B., lee H-S and Brockhoff P.B. (2010). Estimation of
  the Thurstonian model for the 2-AC protocol. Manuscript for Food
  Quality and Preference.
}

\author{Rune Haubo B Christensen}
\seealso{
\code{\link{profile}} and \code{\link{confint}}
}
\examples{

(fm1 <- twoAC(c(2, 2, 6)))
confint(fm1)
confint(fm1, type = "Wald")

pr1 <- profile(fm1)
confint(pr1)

pr1 <- profile(fm1, alpha = 1e-5)
par(mfrow = c(2,2))
plot(pr1)
plot(pr1, Log = FALSE, relative = TRUE)
plot(pr1, Log = TRUE, relative = TRUE)
plot(pr1, Log = TRUE, relative = FALSE)
\dontshow{
## Test stability of CIs:
ci <- c(confint(fm1), confint(fm1, type = "Wald"))
## ci2 := dput(ci)
ci2 <- c(-0.27098319957656, 1.85925113345333, -0.108182667530717, -0.287401447110179,
0.940127921188124, 1.83592051925275)
stopifnot(
isTRUE(all.equal(ci, ci2))
)
}

}
\keyword{models}
