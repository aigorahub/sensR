################################################################
## LINKS FOR DOUBLE METHODS:
################################################################

doubleduotrio <- function() {
  doubleduotrio <- binomial()
  doubleduotrio$link <- "Link for the duo-trio double test"
  doubleduotrio$linkinv <- function(eta) {
    ok <- eta > 0 & eta < 20
    eta[eta <= 0] <- 0.25
    eta[eta >= 20] <- 1
    if(sum(ok)) {
      eta.ok <- eta[ok]
      pnorm.eta.2 <- pnorm(eta.ok * sqrt(1/2))
      pnorm.eta.6 <- pnorm(eta.ok * sqrt(1/6))
      eta[ok] <-
        (1 - pnorm.eta.2 - pnorm.eta.6 + 2 * pnorm.eta.2 * pnorm.eta.6)^2
    }
    pmin(pmax(eta, 0.25), 1) ## restrict to [0.25, 1] - just to be sure
  }
  doubleduotrio$mu.eta <- function(eta) {
    ok <- eta > 0 ## no upper limit
    eta[!ok] <- 0
    eta[eta >= 20] <- 1 ## should we include this restriction?
    if(sum(ok)) {
      eta.ok <- eta[ok]
      pnorm.eta.2 <- pnorm(eta.ok * sqrt(1/2))
      pnorm.eta.6 <- pnorm(eta.ok * sqrt(1/6))
      sqrt.2 <- sqrt(1/2)
      sqrt.6 <- sqrt(1/6)
      eta.2 <- eta.ok * sqrt.2
      eta.6 <- eta.ok * sqrt.6
      A <- dnorm(eta.2) * sqrt.2
      B <- dnorm(eta.6) * sqrt.6
      C <- dnorm(eta.2)
      D <- pnorm(eta.6) * sqrt.2
      E <- pnorm(eta.2)
      eta[ok] <- 2 * (1 - pnorm.eta.2 - pnorm.eta.6 + 2 * pnorm.eta.2 * pnorm.eta.6) *
        (- A - B + 2 * (C * D + E * B))
    }
    pmax(eta, 0) ## gradient cannot be negative.
  }
  doubleduotrio$linkfun <- function(mu) {
    eps <- 1e-10
    ok <- mu > 0.25 & mu < 1 - eps
    mu[mu <= 0.25] <- 0
    mu[mu >= 1 - eps] <- Inf
    if(sum(ok)) {
      duotriog <- function(d, p) doubleduotrio$linkinv(d) - p
      mu[ok] <- sapply(mu[ok], function(mu) {
        uniroot(f=duotriog, interval=c(0, 16), p=mu)$root })
    }
    pmax(mu, 0) ## delta cannot be negative
  }
  doubleduotrio
}

doublethreeAFC <- function ()
{
  doublethreeAFC <- binomial()
  doublethreeAFC$link <- "Link for the 3-AFC double test"
  doublethreeAFC$linkinv <- function(eta) {
    ok <- eta > 0 & eta < 9
    eta[eta <= 0] <- 1/9
    eta[eta >= 9] <- 1
    if(sum(ok)) {
      threeAFCg <- function(x, d) dnorm(x - d) * pnorm(x)^2
      eta[ok] <- sapply(eta[ok], function(eta) {
        (integrate(threeAFCg, -Inf, Inf, d = eta)$value)^2 })
    }
    pmin(pmax(eta, 1/9), 1) ## restrict to [1/9, 1] - just to be sure
  }
  doublethreeAFC$mu.eta <- function(eta) {
    ok <- eta > 0 & eta < 9
    eta[eta <= 0] <- 0
    eta[eta >= 9] <- 0
    ### Note: integration is not reliable for eta > 9
    if(sum(ok)) {
      threeAFCg <- function(x, d) dnorm(x - d) * pnorm(x)^2
      threeAFCgd <- function(x, d)
        (x - d) * dnorm(x - d) * pnorm(x)^2
      eta[ok] <- sapply(eta[ok], function(eta) {
        2 * integrate(threeAFCg, -Inf, Inf, d = eta)$value *
          integrate(threeAFCgd, -Inf, Inf, d = eta)$value})
    }
    pmax(eta, 0) ## gradient cannot be negative.
  }
  doublethreeAFC$linkfun <- function(mu) {
    eps <- 1e-8
    ok <- mu > 1/9 & mu < 1 - eps
    mu[mu <= 1/9] <- 0
    mu[mu >= 1 - eps] <- Inf
    if(sum(ok)) {
      threeAFCg2 <- function(d, p) doublethreeAFC$linkinv(d) - p
      mu[ok] <- sapply(mu[ok], function(mu)
        uniroot(threeAFCg2, c(0, 9), p = mu)$root)
    }
    pmax(mu, 0)
  }
  doublethreeAFC
}

doubletriangle <- function()
{
  doubletriangle <- binomial()
  doubletriangle$link <- "Link for the triangle double test"
  doubletriangle$linkinv <- function(eta) {
    ok <- eta > 0 & eta < 20
    eta[eta <= 0] <- 1/9
    eta[eta >= 20] <- 1
    if(sum(ok))
      eta[ok] <-
      (pf(q=3, df1=1, df2=1, ncp=eta[ok]^2*2/3, lower.tail=FALSE))^2
    pmin(pmax(eta, 1/9), 1) ## restrict to [1/9, 1] - just to be sure
  }
  # doubletriangle$mu.eta <- function(eta) {
  #   ok <- eta > 0 & eta < 20
  #   eta.1 <- eta
  #   eta.1[eta.1 <= 0] <- 1/9
  #   eta.1[eta.1 >= 20] <- 1
  #   if(sum(ok))
  #     eta.1[ok] <-
  #     2*pf(q=3, df1=1, df2=1, ncp=eta.1[ok]^2*2/3, lower.tail=FALSE)
  #   eta.1 <- pmin(pmax(eta.1, 1/9), 1) ## restrict to [1/9, 1] - just to be sure
  #   eta[eta <= 0] <- 0
  #   eta[eta >= 20] <- 0
  #   if(sum(ok)) {
  #     d <- eta[ok]
  #     eta[ok] <- eta.1 * sqrt(2/3) * dnorm(d/sqrt(6)) *
  #       (pnorm(d/sqrt(2)) - pnorm(-d/sqrt(2)))
  #   }
  #
  #   pmax(eta, 0) ## gradient cannot be negative.
  # }
  doubletriangle$mu.eta <- function(eta) {
    ok <- eta > 0 & eta < 20
    eta[eta <= 0] <- 1/9
    eta[eta >= 20] <- 1
    if(sum(ok)) {
      d <- eta[ok]
      eta[ok] <- 2*pf(q=3, df1=1, df2=1, ncp=d^2*2/3, lower.tail=FALSE) *
          sqrt(2/3) * dnorm(d/sqrt(6)) *
              (pnorm(d/sqrt(2)) - pnorm(-d/sqrt(2)))
    }

    pmax(eta, 0) ## gradient cannot be negative.
  }
  doubletriangle$linkfun <- function(mu) {
    eps <- 1e-8
    ok <- mu > 1/9 & mu < 1 - eps
    mu[mu <= 1/9] <- 0
    mu[mu >= 1 - eps] <- Inf
    if(sum(ok)) {
      triangleg2 <- function(d, p) doubletriangle$linkinv(d) - p
      mu[ok] <- sapply(mu[ok], function(mu)
        uniroot(triangleg2, c(0, 15), p = mu)$root)
    }
    mu # FIXME: pmax(mu, 0)
  }
  doubletriangle
}

doubletetrad <- function()
{
  doubletetrad <- binomial()
  doubletetrad$link <- "Link for the unspecified tetrad test"
  doubletetrad$linkinv <- function(eta) {
    eps <- 1e-8
    ok <- eta > eps & eta < 9
    eta[eta <= eps] <- 1/9
    eta[eta >= 9] <- 1
    if(sum(ok)) {
      tetrads.fun <- function(z, delta)
        dnorm(z) * (2 * pnorm(z) * pnorm(z - delta) -
                      pnorm(z - delta)^2)
      eta[ok] <- sapply(eta[ok], function(eta) {
        (1 - 2*integrate(tetrads.fun, -Inf, Inf, delta=eta)$value)^2 })
    }
    pmin(pmax(eta, 1/9), 1) ## restrict to [1/9, 1] - just to be sure
  }
  doubletetrad$mu.eta <- function(eta) {
    eps <- 1e-8
    ok <- eta > eps & eta < 9
    eta[eta <= eps] <- 0
    eta[eta >= 9] <- 0
    if(sum(ok)) {
      Linkinv <- function(eta) {
        tetrads.fun <- function(z, delta)
          dnorm(z) * (2 * pnorm(z) * pnorm(z - delta) -
                        pnorm(z - delta)^2)
        sapply(eta, function(eta) {
          (1 - 2*integrate(tetrads.fun, -Inf, Inf, delta=eta)$value)^2 })
      }
      eta[ok] <- sapply(eta[ok], function(eta) grad(Linkinv, eta))
      ### FIXME: Could probably do the integration by hand here.
    }
    pmax(eta, 0) ## gradient cannot be negative.
  }
  doubletetrad$linkfun <- function(mu) {
    eps <- 1e-8 ## What is the right eps here?
    ok <- mu > 1/9 & mu < 1 - eps
    mu[mu <= 1/9] <- 0
    mu[mu >= 1 - eps] <- Inf
    if(sum(ok)) {
      doubletetrads <- function(d, p) doubletetrad$linkinv(d) - p
      mu[ok] <- sapply(mu[ok], function(mu)
        uniroot(doubletetrads, c(0, 9), p = mu)$root)
    }
    pmax(mu, 0)
  }
  doubletetrad
}

doubletwoAFC <- function() {
  doubletwoAFC <- binomial()
  doubletwoAFC$link <- "Link for the 2-AFC double test"
  doubletwoAFC$linkinv <- function(eta) {
    ok <- eta > 0
    eta[!ok] <- 0.25
    eta[ok] <- (pnorm(eta[ok] / sqrt(2)))^2
    pmin(pmax(eta, 0.25), 1) ## restrict to [0.5, 1] - just to be sure
  }
  doubletwoAFC$mu.eta <- function(eta) {
    eta.linkinv <- eta
    ok <- eta > 0
    eta[!ok] <- 0
    eta.linkinv[!ok] <- 0.25
    if(any(ok)) {
      sqrt.2 <- sqrt(1/2)
      eta.linkinv[ok] <- pnorm(eta.linkinv[ok] / sqrt(2))
      eta[ok] <- sqrt(2) * eta.linkinv[ok] * dnorm(eta[ok] * sqrt.2)
    }
    pmax(eta, 0) ## gradient cannot be negative
  }
  doubletwoAFC$linkfun <- function(mu) {
    ok <- mu > 0.25 & mu < 1
    mu[mu <= 0.25] <- 0
    mu[mu >= 1] <- Inf
    # mu[ok] <- 2 * qnorm(mu[ok])^2
    # pmax(mu, 0) ## delta cannot be negative
    if(sum(ok)) {
      twoafcs <- function(d, p) doubletwoAFC$linkinv(d) - p
      mu[ok] <- sapply(mu[ok], function(mu)
        uniroot(twoafcs, c(0, 9), p = mu)$root)
    }
    pmax(mu, 0)
  }
  doubletwoAFC
}
