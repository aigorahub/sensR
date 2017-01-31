\name{posthoc}
\alias{posthoc}
\alias{posthoc.dprime_compare}
\alias{posthoc.dprime_test}
\title{
  Post-hoc estimates and tests for multiple discrimination experiments.
}
\description{
  This function provides estimates and p-values for post-hoc tests such
  as pairwise comparisons. p-values are (by default) adjusted for multiplicity.
}
\usage{

posthoc(x, ...)

\method{posthoc}{dprime_compare}(x, alpha = 0.05,
    test = c("pairwise", "common", "base", "zero"), base = 1,
    alternative = c("two.sided", "less", "greater"),
    statistic = c("likelihood", "Wald"),
    padj.method = c("holm", "bonferroni", "none"), ...)

\method{posthoc}{dprime_test}(x, alpha = 0.05,
    test = c("pairwise", "common", "base", "zero"), base = 1,
    alternative = c("two.sided", "less", "greater"),
    statistic = c("likelihood", "Wald"),
    padj.method = c("holm", "bonferroni", "none"), ...)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
    an object of class \code{\link{dprime_compare}} or
    \code{\link{dprime_test}}.
  }
  \item{alpha}{
    the significance level for tests and confidence intervals.
  }
  \item{test}{
    the type of post-hoc tests performed. Se the details section for
    further details.
  }
  \item{base}{
    when \code{test = "base"}, the experiment against which to provide
    pairwise comparisons.
  }
  \item{alternative}{
    direction of the hypothesis test.
  }
  \item{statistic}{
    The test statistic used - currently there is only partial support
    for \code{statistic = "likelihood"}.
  }
  \item{padj.method}{
    controls the method by which p-values are adjusted for
    multiplicity. Any one of the values in \code{p.adjust.methods}
    (currently "holm" "hochberg"
    "hommel" "bonferroni" "BH" "BY" "fdr" "none") may be specified,
    cf. \code{\link{p.adjust}}.
  }
  \item{\dots}{
    currently not used.
  }
}
\details{
  The \code{test} argument specifies the type of test
    performed. \code{"pairwise"} performs all pairwise comparisons and
    produces a compact letter display indicating groups of experiments
    that different/not-different. \code{"common"} tests, for each
    experiment in turn, if the by-experiment d-prime is different from
    a common d-prime computed from the remaining
    experiments. \code{"base"} provides pairwise comparisons to a
    single experiment indicated by the separate argument \code{base}. If
    \code{test = "zero"} all d-primes are tested versus zero. As a final
    option a numeric value can be supplied, e.g. \code{test = 1} in
    which case all d-primes are tested versus one. Note that
    \code{test = 0} gives the same test as \code{test = "zero"}.

    When \code{test = "pairwise"} a compact letter display is provided and
    it is determined from the p-values \emph{after} adjustment of these for
    multiplicity.

    The \code{dprime_compare} and \code{dprime_test} methods a have
    (common) print method.
}
\value{
  an object of class \code{c(paste0("posthoc.", class(x)), class(x))}
  with the following elements from the original object, \code{x} and :
  \item{posthoc}{
    coefficient table for the post-hoc tests.
  }
  \item{test}{
    the value of the \code{test} argument.
  }
  \item{alternative}{
    the value of the \code{alternative} argument.
  }
  \item{padj.method}{
    the method used to adjust p-values with.
  }
  \item{base}{
    the value of the \code{base} argument.
  }
  \item{posthoc.stat}{
    name of the statistic for the post-hoc tests.
  }
  \item{Letters}{
    if \code{test = "pairwise"} the compact letter display, otherwise
    \code{NULL}.
  }
  \item{dprime0}{
    unless \code{test = "pairwise"} or \code{"common"} the value of
    d-prime under the null hypothesis.
  }
}
\author{
  Rune Haubo B Christensen
}
\seealso{
  \code{\link{dprime_test}}, \code{\link{dprime_table}},
  \code{\link{dprime_compare}}.
}
\examples{

## Make some fake data:
n <- rep(40, 4)
x <- c(25, 25, 30, 35)
protocol <- c("triangle", "duotrio", "threeAFC", "twoAFC")
## Look at the data table with d-primes etc.:
dprime_table(x, n, protocol)

## 'any differences' test:
## ML estimation and test with likelihood statistic:
(dpc <- dprime_compare(x, n, protocol))

posthoc(dpc, alpha=.1) ## test="pairwise"

## Test if each d' is different from the common d' estimated from the
## remaining experiments:
posthoc(dpc, test="common")

## Test if d' from experiment 2 is different from the others (with
## adjustment for multiplicity):
posthoc(dpc, test="base", base=2)

## Test if each d' is different from 2 (with Bonferroni adjustment for
## multiplicity) using the Wald statistic:
posthoc(dpc, test=2, stat="Wald", padj.method="bonferroni")

}
\keyword{models}

