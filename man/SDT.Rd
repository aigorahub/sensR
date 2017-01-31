\name{SDT}
\alias{SDT}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{Signal Detection Theory Computation of d-prime}
\description{
  The function computes d-prime for any 2 x J table where J >= 2 for the
  "yes--no" or "A-Not A" experiment using the Signal Detection Theory
  (SDT) algorithm to compute J-1 d-prime's. The algorithm is also called
  the "empirical probit transform". The function also provides the "logit"
  counterpart.
}
\usage{
SDT(tab, method = c("probit", "logit"))
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{tab}{A 2 x J table with true class relation in rows (only two
    true classes) and the J-class response in columns}
  \item{method}{should the empirical probit or logit transform be computed?}
}
%% \details{
%%   ~~ If necessary, more details than the description above ~~
%% }
\value{
  A (J-1) x 3 matrix. The first two columns contains the z-transform of
  the Hit rate and the False Alarm rate respectively---ready to plot
  along with the empirical ROC curve. The third column contains the
  estimated d-primes.
}
\references{MacMillan , A. N. and Creelman, C. D (2005) Detection Theory
A User's Guide. Lawrence Elbaum Associates, Inc. 2nd edition.}
\author{Rune Haubo B Christensen}
%% \note{ ~~further notes~~
%%
%%  ~Make other sections like Warning with \section{Warning }{....} ~
%% }
%% \seealso{ ~~objects to See Also as \code{\link{help}}, ~~~ }
\examples{
### Design table:
## 8  "yes"-responses to yes-samples
## 1  "yes"-responses to no-samples
## 17 "no"-response to yes-samples
## 24 "no"-responses to no-samples
## Note that response-class is columnwise and true-class is rowwise:
(mat <- rbind(c(8, 17),
              c(1, 24)))
SDT(mat, "logit")
SDT(mat, "probit")

## compare to AnotA():
m1 <- AnotA(8, 25, 1, 25)
m1

## Compute d-prime 'by hand':
## Hit rate and False alarm rates:
H <- 8/(8+17)
FA <- 1/(1+24)
zH <- qnorm(H)
zFA <- qnorm(FA)
## d-prime:
zH - zFA  # d'

\dontshow{
  ## Testing correctness of examples:
  x <- as.vector(SDT(mat, "logit"))
  ## xx := dput(x)
  xx <- c(-0.75377180237638, -3.17805383034795, 2.42428202797157)
  stopifnot(isTRUE(all.equal(x, xx)))

  y <- as.vector(SDT(mat, "probit"))
  ## yy := dput(y)
  yy <- c(-0.467698799114508, -1.75068607125217, 1.28298727213766)
  stopifnot(isTRUE(all.equal(y, yy)))

  ## Test equality of all three d-prime computation methods:
  yy2 <- as.vector(coef(m1))
  yy3 <- zH - zFA
  stopifnot(isTRUE(all.equal(yy[3], yy2, yy3)))

}

## Multi-response-class example (odor example from MacMillan and
## Creelman, 2005)
(odor <- matrix(c(112, 112, 72, 53, 22, 4, 7, 38, 50, 117, 101, 62), 2,
               byrow = TRUE))
obj <- SDT(odor)
ROC(obj[3,3])
\dontshow{
  ## x := dput(as.vector(obj))
  x <- c(-0.528239188042123, 0.246450738699479, 0.804110201686501, 1.48077452682481,
  2.30203260833608, -2.08210003525935, -1.17498679206609, -0.664036943094355,
  0.164505352668462, 0.972771929727746, 1.55386084721722, 1.42143753076557,
  1.46814714478086, 1.31626917415635, 1.32926067860833)
  stopifnot(isTRUE(all.equal(x, as.vector(obj))))

}
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{models}
%% \keyword{ ~kwd2 }% __ONLY ONE__ keyword per line
