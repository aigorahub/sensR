\name{plot.discrim}
\alias{plot.discrim}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{ Plot function for discrim objects}
\description{
  This function plots the latent distributions of sensory intensity
  corresponding to the items or products tested in the discrimination
  test.   
}
\usage{
\method{plot}{discrim}(x, main = TRUE, length = 1000, ...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{The \code{discrim} object whose latent distributions are to
    be plotted}
  \item{main}{include an automatically generated title on the plot?
    Default is \code{TRUE}}
  \item{length}{the length of the vectors to be plotted. Longer vectors
    gives more smooth curves.}
  \item{\dots}{additional arguments to \code{plot} and \code{lines}}
}
%% \details{
%%   ~~ If necessary, more details than the description above ~~
%% }
%% \value{
%%   The function produce the plot and does not return any value.
%% }
%% \references{ ~put references to the literature/web site here ~ }
\author{Rune Haubo B Christensen}
%% \note{ ~~further notes~~ 
%% 
%%  ~Make other sections like Warning with \section{Warning }{....} ~
%% }
%% \seealso{ ~~objects to See Also as \code{\link{help}}, ~~~ }
\examples{
## Generate discrim objects to be plotted:
fm1 <- discrim(10, 15, method = "threeAFC")
fm2 <- discrim(10, 15, method = "triangle")
par(mfrow=c(2,1)) ## Split plotting window in two
## Plot the distributions of sensory intensity for the two objects
## and increase the line width
plot(fm1, lwd=2) 
plot(fm2, lwd=2)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{hplot}
%% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
