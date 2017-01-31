\name{dprime_table}
\alias{dprime_table}
\title{
  Summary table of several discrimination experiments using the
  simple-binomial protocols (Duo-Trio, Triangle, Tetrad, 2-AFC and 3-AFC)
}
\description{
  This function provides a summary table with the following quantities:
  no. correct trials (\code{correct}), total number of trials
  (\code{total}), the
  protocol (\code{protocol}), probability of a correct answer
  (\code{pHat}), standard
  error of pHat (\code{se.pHat}), d-prime (\code{dprime}), and standard
  error of d-prime (\code{se.dprime}).
}
\usage{
dprime_table(correct, total, protocol, restrict.above.guess = TRUE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{correct}{
    a numeric vector of the number of correct answers; one element for
    each test.
  }
  \item{total}{
    a numeric vector of the total number of trials; one element for each
    test.
  }
  \item{protocol}{
    a character vector or factor naming the protocol used; one element
    for each test. Currently the following protocols are supported:
    \code{"triangle", "duotrio", "threeAFC", "twoAFC", "tetrad"}.
  }
  \item{restrict.above.guess}{
    controls if \code{pHat} should be restricted at or
    above the guessing probability for the given protocol. This also
    affects \code{se.pHat}. Note that \code{dprime} is zero and
    \code{se.dprime} is \code{NA} when
    \code{pHat} is at or below the guessing probability of the given
    protocol.
  }
}
\details{
  The vectors \code{correct}, \code{total} and \code{protocol} have to
  be of the same length.
}
\value{
  a data.frame with columns:
  \item{correct}{
    numeric vector of no. correct.
  }
  \item{total}{
    numeric vector of no. trials.
  }
  \item{protocol}{
    character vector naming the protocols used.
  }
  \item{pHat}{
    Estimate of the probability of correct answers.
  }
  \item{se.pHat}{
    standard error of \code{pHat}.
  }
  \item{dprime}{
    estimate of d-prime.
  }
  \item{se.dprime}{
    standard error of \code{dprime}.
  }
}
\author{
  Rune Haubo B Christensen
}
\seealso{
  \code{\link{dprime_compare}}, \code{\link{dprime_test}},
  \code{\link{posthoc.dprime_compare}}.
}
\examples{

n <- rep(40, 4)
x <- c(25, 25, 30, 35)
protocol <- c("triangle", "duotrio", "threeAFC", "twoAFC")
dprime_table(x, n, protocol)

}
\keyword{models}

