\name{clm2twoAC}
\alias{clm2twoAC}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Extract 2-AC coefficient table from a cumulative link model}
\description{
  The Thurstonian model for the 2-AC protocol can be formulated as a
  cumulative link model (see the references). This function extracts the
  2-AC model parameter
  estimates, standard errors, z-value and p-values from a cumulative
  link (mixed) model fitted with \code{\link[ordinal]{clm}} or
  \code{\link[ordinal]{clmm}} from package \code{ordinal}.
}
\usage{

clm2twoAC(object, ...)

}
\arguments{
  \item{object}{ a \code{\link[ordinal]{clm}} or
    \code{\link[ordinal]{clmm}} object
    }
  \item{\dots}{not currently used.}
}
\seealso{
  \code{\link{twoAC}}, \code{\link{twoACpwr}}
}
\value{
  A \code{data.frame} with the coefficient table. The two first rows
  contain the estimates of \code{tau} and \code{d.prime} while the
  remaining rows contain optional regression variables for
  \code{d.prime}.
}
\references{
  Christensen R.H.B., Lee H-S and Brockhoff P.B. (2011). Estimation of
  the Thurstonian model for the 2-AC protocol. Submitted to Food
  Quality and Preference.
}

\author{Rune Haubo B Christensen}

\examples{

## Example of a simple 2-AC model. First the conventional way:
twoAC(c(2, 2, 6))
\dontshow{
## Testing stability of twoAC function:
fm1 <- twoAC(c(2, 2, 6))
## str(fm1)
nm <- c("coefficients", "stat.value", "p.value", "confint", "vcov", "logLik")
b <- as.vector(unlist(fm1[nm]))
## b2 := dput(b)
b2 <- c(0.415972626828703, 0.774259536071287, 0.26743108470048, 0.541673720311043,
1.44671786297779, 0.147975959384948, -0.27098319957656, 1.85925113345333,
0.0715193850640751, 0.0216722514182911, 0.0216722514182911, 0.293410419275606,
-9.50270539233235)
stopifnot(isTRUE(all.equal(b, b2)))
}
## The using a cumulative link model (clm from package ordinal):
if(require(ordinal)) {
    response <- gl(3,1)
    fit.clm <- clm(response ~ 1, weights = c(2, 2, 6), link = "probit")
    clm2twoAC(fit.clm)
    ## Alternatively we could get estimates and standard errors "by hand":
    tab <- coef(summary(fit.clm))
    theta <- tab[,1]
    (tau <- (theta[2] - theta[1])/sqrt(2))
    (d.prime <- (-theta[2] - theta[1])/sqrt(2))
    VCOV <- vcov(fit.clm)
    (se.tau <- sqrt((VCOV[1,1] + VCOV[2,2] - 2*VCOV[2,1])/2))
    (se.d.prime <- sqrt((VCOV[1,1] + VCOV[2,2] + 2*VCOV[2,1])/2))

    ## Extended example with a regression model for d.prime
    ## (see the referenced paper for details):
    n.women <- c(2, 2, 6)*10
    n.men <- c(1, 2, 7)*10
    wt <- c(n.women, n.men)
    response <- gl(3,1, length = 6)
    gender <- gl(2, 3, labels = c("women", "men"))
    fm2 <- clm(response ~ gender, weights = wt, link = "probit")
    clm2twoAC(fm2)
}
\dontshow{
  ## Test equality of hand calculations, twoAC and clm2twoAC:
  if(require(ordinal)){
    b <- clm2twoAC(fit.clm)
    b2 <- c(tau, d.prime, se.tau, se.d.prime)
    stopifnot(
    isTRUE(all.equal(unlist(b[, 1:2]), b2, c(coef(fm1)), check.attributes=FALSE))
    )
    ## Test stability of clm2twoAC results:
    tab <- unlist(clm2twoAC(fm2)[, 1:3])
    ## tab2 := dput(tab)
    tab2 <- c(0.467000059796145, 0.78987850607203, 0.453312459391865,
    0.0670001441456634, 0.170212706442409, 0.246277460374108, 6.9701351504685,
    4.64053784574117, 1.84065751978788)
    stopifnot(
    isTRUE(all.equal(tab, tab2, check.attributes=FALSE))
    )
  }
}

}
\keyword{models}% at least one, from doc/KEYWORDS
