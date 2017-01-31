\name{rescale}
\alias{rescale}
%% \alias{print.rescale}
\alias{psyfun}
\alias{psyinv}
\alias{psyderiv}
\alias{pc2pd}
\alias{pd2pc}
\title{Transform or rescale between pc, pd and d-prime for sensory
  discrimination protocols}
\description{Transforms or rescales estimates and optionally standard
  errors between the three levels at which a sensory difference is
  measured: pc (proportion of correct answers), pd (proportion of
  discriminators) and d-prime. \code{rescale} is the main function and
  only one of pc, pd or d-prime should be given as argument --- values
  for the remaining two scales will be computed.

  A number of auxiliary functions are also provided:
  \describe{
    \item{\code{psyfun}}{implements the psychometric functions and maps
      from d-prime to pc}
    \item{\code{psyinv}}{implements the inverse psychometric functions
      and maps from pc to d-prime}
    \item{\code{psyderiv}}{implements the derivative of the psychometric
      functions}
    \item{\code{pc2pd}}{maps from pc to pd}
    \item{\code{pd2pc}}{maps from pd to pc}
    }
} 
\usage{

rescale(pc, pd, d.prime, std.err, 
        method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
        "triangle")) 

psyfun(d.prime, method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
       "triangle")) 

psyinv(pc, method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
       "triangle")) 

psyderiv(d.prime, method = c("duotrio", "tetrad", "threeAFC", "twoAFC",
         "triangle")) 

pc2pd(pc, Pguess)

pd2pc(pd, Pguess)

}
\arguments{
  \item{pc}{the proportion of correct answers; a numerical vector
    between 0 and 1}
  \item{pd}{the proportion of discriminators; a numerical vector between
    0 and 1} 
  \item{d.prime}{the sensory difference on the d-prime scale; a
    non-negative numerical vector.} 
  \item{std.err}{optional numerical vector of standard errors of the
    same length as the either of \code{pc}, \code{pd} or
    \code{d.prime}. Negative values are not allowed, but values may be
    \code{NA}}  
  \item{method}{the sensory discrimination protocol for which the
    results should apply}
  \item{Pguess}{the guessing probability implied by the protocol; a
    numeric scalar between 0 and 1}
}
\details{
  The \code{rescale} function is based on the fact that once the
  protocol and one of 
  pc, pd and d-prime is known, the other two can be computed. The same
  applies to the standard errors of these parameters.

  Standard errors are optional, but if they are supplied, the length of
  the \code{std.err} argument has to match the length of \code{pc},
  \code{pd} or \code{d.prime} whichever is given.

  A \code{print} method is implemented for \code{rescale} objects.
}
\value{
  For \code{rescale} an object of class \code{rescale} with elements
  \item{coefficients}{a \code{data.frame} with values of \code{pc},
    \code{pd} and \code{d.prime} corresponding to the input}
  \item{std.err}{if standard errors are given trough the \code{std.err}
    argument a \code{data.frame} of the same size and shape as
    \code{coefficients} with standard errors. Otherwise missing.}
  \item{method}{the sensory discrimination protocol for which the
    results apply}

  For \code{psyfun}, \code{psyinv}, \code{psyderiv}, \code{pc2pd} and
  \code{pd2pc} a numerical vector of the same length as the first
  argument with appropriate contents.
}

\author{Rune Haubo B Christensen}

\examples{

## suppose 15 out of 20 are observed in a duo-trio experiment, then
## the estimated probability of correct a answer is
(pc <- 15/20)
## The standard error of this estimate is
(se.pc <- sqrt(pc * (1 - pc) / 20))
## The corresponding estimate of proportion of discriminators (pd) and
## d-prime with associated standard errors are:
rescale(pc = pc, std.err = se.pc, method = "duotrio")

## Can also do
rescale(pd = c(.6,.7), std.err = c(.2, NA))
psyfun(2, method = "triangle")
psyinv(0.8, method = "twoAFC")
psyderiv(2, method = "duotrio")
pc2pd(0.7, 1/2)
pd2pc(0.3, 1/3)

}
\keyword{models}% at least one, from doc/KEYWORDS

