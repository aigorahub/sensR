\name{plot.samediff}
\alias{plot.samediff}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Plot function for samediff objects}
\description{
  This function plots the latent distributions of sensory intensity
  corresponding to the items or products tested in the discrimination
  test.
}
\usage{
\method{plot}{samediff}(x, main = TRUE, length = 1000,
           limits, fig = TRUE, ...)

}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{The \code{samediff} object whose latent distributions are to
    be plotted}
  \item{main}{include an automatically generated title on the plot?
    Default is \code{TRUE}}
  \item{length}{the length of the vectors to be plotted. Longer vectors
    gives more smooth curves, but can take a little time.}
  \item{limits}{optional limits on the x-axis; vector of length two.}
  \item{fig}{logical: Should the function create the plot? Defaults to \code{TRUE}.}
  \item{\dots}{additional arguments to \code{plot} and \code{lines}}
}
%% \details{
%%   ~~ If necessary, more details than the description above ~~
%% }
\value{
  If \code{fig = TRUE}, the function will produce the plot. The function
  invisibly returns a data.frame with elements
  \item{z}{values for the x-axis of lenght \code{length}.}
  \item{base.dist}{y-values for the base distribution of same-samples,
    ie. a standard normal distribution}
  \item{delta.dist}{y-values for the distribution of different-samples,
    ie. a normal distribution centred at \code{delta} with unit
    variance.}
  This facilitates later plotting and changing the appearance of the
  plot. 
}
%% \references{ ~put references to the literature/web site here ~ }
\author{Rune Haubo B Christensen}
%% \note{ ~~further notes~~ 
%% 
%%  ~Make other sections like Warning with \section{Warning }{....} ~
%% }
%% \seealso{ ~~objects to See Also as \code{\link{help}}, ~~~ }
\examples{
## Make same-diff object:
sadi <- samediff(8, 5, 4, 9)
## Plot distributions of sensory intensity:
plot(sadi)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{hplot}
%% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
