library(sensR)

## change discrim
discrim(correct=13, total=30, method="triangle")
discrim(correct=13, total=30, method="triangle", double = TRUE)

## change findcr
findcr(sample.size = 20, alpha = 0.05, p0 = 0.5)
## example from the J.Bi paper
findcr(sample.size = 100, alpha = 0.05, p0 = 0.5)
findcr(sample.size = 100, alpha = 0.05, p0 = 1/4) ## critical value for the double duo-trio method
findcr(sample.size = 100, alpha = 0.05, p0 = 1/9) ## critical value for the double triangle method


## discrimPwr
discrimPwr(pdA = 0.5, sample.size = 20, alpha = 0.05, pGuess = 1/2)
## example from the J.Bi paper 
pdA <- pc2pd(0.35, 1/4)
discrimPwr(pdA = pdA, sample.size = 100, alpha = 0.05, pGuess = 1/4, statistic = "exact")
discrimPwr(pdA = pdA, sample.size = 100, alpha = 0.05, pGuess = 1/4, statistic = "normal")

stopifnot(all.equal(discrimPwr(pdA = pdA, sample.size = 100, alpha = 0.05, pGuess = 1/4, statistic = "normal"),
          0.7268466, tol=1e-5))

## test double triangle
discrim(10, 15, method = "twoAFC")
discrim(10, 15, method = "twoAFC", double = TRUE, statistic = "likelihood")
findcr(sample.size = 15, alpha = 0.05, p0 = 1/4) ## critical value for the double twoAFC method

discrim(35,100, method="duotrio")
psyinv(0.5915, method="duotrio")
psyinv(0.5915, method="twoAFC")
psyinv(0.35, method="twoAFC", double = TRUE)
psyfun(1.060117, method = "duotrio")
psyinv(0.5915, method="duotrio")

k <- (1 + psyfun(1.060117, method = "duotrio"))/(4*psyfun(1.060117, method = "duotrio"))
k


## check psyfun for the twoAFC double method
psyfun(1.060117, method = "twoAFC")
stopifnot(all.equal(psyinv(0.773257, method="twoAFC"),1.060117, tol=1e-5))
psyderiv(1.060117, method = "twoAFC")


psyfun(1.060117, method = "twoAFC", double = TRUE)
psyinv(0.5979263, method="twoAFC", double = TRUE)

stopifnot(all.equal(psyfun(1.060117, method = "twoAFC", double = TRUE),
                    0.5979263, tol=1e-6))

## check psyderiv
## g' =2 * f * f'
stopifnot(all.equal(psyderiv(1.060117, method = "twoAFC", double = TRUE),
                    2 * psyfun(1.060117, method = "twoAFC") * psyderiv(1.060117, method = "twoAFC"), 
                    tol = 1e-6))

stopifnot(all.equal(psyderiv(2, method = "twoAFC", double = TRUE),
                    2 * psyfun(2, method = "twoAFC") * psyderiv(2, method = "twoAFC"), 
                    tol = 1e-6))

#####################################################################
## check discrim for the double methods
#####################################################################

## check twoAFC
discrim(10, 15, method = "twoAFC")
discrim(10, 15, method = "twoAFC", double = TRUE)
## check pc in the discrim table
stopifnot(all.equal(psyfun(1.2758, method = "twoAFC", double = TRUE), 0.6667, tol=1e-4))

## check threeAFC          
discrim(10, 15, method = "threeAFC")
discrim(10, 15, method = "threeAFC", double = TRUE)
stopifnot(all.equal(psyfun(1.7316, method = "threeAFC", double = TRUE),
                    0.6667, tol=1e-4))

## check duotrio
discrim(10, 15, method = "duotrio")
discrim(10, 15, method = "duotrio", double = TRUE)
stopifnot(all.equal(psyfun(2.4764, method = "duotrio", double = TRUE), 0.6667, tol=1e-4))

## check double triangle
discrim(10, 15, method = "triangle")
discrim(10, 15, method = "triangle", double = TRUE)
stopifnot(all.equal(psyfun(3.2497, method = "triangle", double = TRUE), 0.6667, tol=1e-4))

############################################################################
## check from the J.Bi paper 
############################################################################
psyinv(0.35, method = "duotrio", double = TRUE)
## conventional method
stopifnot(all.equal(psyfun(1.06, method = "duotrio"), 0.5915, tol=1e-4))
stopifnot(all.equal(psyfun(1.06, method = "duotrio", double = TRUE), 0.35, tol=1e-3))

## check the standard error of d prime for double methods
#############################
## for duo-trio
(ex1 <- discrim(35, 100, method="duotrio"))
(ex1.doub <- discrim(35, 100, method="duotrio", double = TRUE))
stopifnot(all.equal(pc2pd(0.35, 1/4),coefficients(ex1.doub)[2,1], tol=1e-5))
stopifnot(all.equal(1.060826,coefficients(ex1.doub)[3,1], tol=1e-5))

## check B.star
stopifnot(all.equal(0.35*(1-0.35)/(psyderiv(coefficients(ex1.doub)[3,1], 
                                            method = "duotrio", double = TRUE))^2,
          0.6726*10.409, tol=1e-3))
stopifnot(all.equal(coefficients(ex1.doub)[3,2]^2*100, 0.6726*10.409, tol=1e-2))
stopifnot(all.equal(coefficients(ex1.doub)[3,2]^2, 0.07, tol=1e-3))
stopifnot(all.equal(psyfun(coefficients(ex1.doub)[3,1], 
                           method = "duotrio", 
                           double = TRUE) * (1 - psyfun(coefficients(ex1.doub)[3,1], 
                                                        method = "duotrio", 
                                                        double = TRUE)) / psyderiv(coefficients(ex1.doub)[3,1],
                                                                                    method = "duotrio",
                                                                                    double = TRUE)^2/100,
                    0.07, tol = 1e-3))


## check derivatives
stopifnot(all.equal(psyderiv(coefficients(ex1.doub)[3,1], method = "duotrio", double = TRUE),
         2*psyfun(coefficients(ex1.doub)[3,1], 
                  method = "duotrio") * psyderiv(coefficients(ex1.doub)[3,1], 
                                               method = "duotrio"),
         tol=1e-6))
####################################
## for threeAFC
(ex2 <- discrim(20,25, method = "threeAFC"))
B.ex2 <-  3.4954 ## from the paper
stopifnot(all.equal(coefficients(ex2)[3,2]^2*25, B.ex2, tol=1e-2))

(ex2.doub <- discrim(20,25, method = "threeAFC", double = TRUE))
stopifnot(all.equal(pc2pd(0.8, 1/9),coefficients(ex2.doub)[2,1], tol=1e-5))
stopifnot(all.equal(psyfun(2.189, method = "threeAFC", double = TRUE), 0.8, tol=1e-3))

## check B.star
var.d <- 0.14
B.ex2 <- (psyfun(2.189, method = "threeAFC") * (1 - psyfun(2.189, 
                                              method = "threeAFC")))/ (psyderiv(2.189, method = "threeAFC"))^2
k <- (1 + psyfun(coefficients(ex2.doub)[3,1], method = "threeAFC"))/(4*psyfun(coefficients(ex2.doub)[3,1], method = "threeAFC"))
stopifnot(all.equal(k*B.ex2/25, coefficients(ex2.doub)[3,2]^2, tol=1e-3))

## check that B.star/N = f*(1-f)/(N*f'^2) is equal to the squared standard error 
stopifnot(all.equal(psyfun(coefficients(ex2.doub)[3,1], 
                           method = "threeAFC", 
                           double = TRUE) * (1 - psyfun(coefficients(ex2.doub)[3,1], 
                                                        method = "threeAFC", 
                                                        double = TRUE)) / psyderiv(coefficients(ex2.doub)[3,1],
                                                                                   method = "threeAFC",
                                                                                   double = TRUE)^2/25,
                    coefficients(ex2.doub)[3,2]^2, tol = 1e-3))
## check that B.star/N = Pc*(1-Pc)/(N*g'^2)
stopifnot(all.equal(coefficients(ex2.doub)[1,1]*(1-coefficients(ex2.doub)[1,1])/(psyderiv(coefficients(ex2.doub)[3,1],
                                                                                          method = "threeAFC",
                                                                                          double = TRUE)^2*25),
                    coefficients(ex2.doub)[3,2]^2, tol=1e-4))

## check derivatives
stopifnot(all.equal(psyderiv(coefficients(ex2.doub)[3,1], method = "threeAFC", double = TRUE),
                    2*psyfun(coefficients(ex2.doub)[3,1], 
                             method = "threeAFC") * psyderiv(coefficients(ex2.doub)[3,1], 
                                                            method = "threeAFC"),
                    tol=1e-6))

####################################################################
## for triangle
(ex3 <- discrim(20,25, method = "triangle"))
B <- 8.144 ## from the paper
B <- (psyfun(coefficients(ex3)[3,1], 
             method = "triangle") * (1 - psyfun(coefficients(ex3)[3,1], 
                                                method = "triangle"))) / psyderiv(coefficients(ex3)[3,1], 
                                                                                  method = "triangle")^2
stopifnot(all.equal(coefficients(ex3)[3,2]^2*25, B, tol=1e-2))

(ex3.doub <- discrim(20,25, method = "triangle", double = TRUE))
B <- (psyfun(coefficients(ex3.doub)[3,1], 
             method = "triangle") * (1 - psyfun(coefficients(ex3.doub)[3,1], 
                                                method = "triangle"))) / psyderiv(coefficients(ex3.doub)[3,1], 
                                                                                  method = "triangle")^2

k <- (1 + psyfun(coefficients(ex3.doub)[3,1], method = "triangle"))/(4*psyfun(coefficients(ex3.doub)[3,1], method = "triangle"))

## TODO check, does not work so far
##(all.equal(k*B/25, coefficients(ex3.doub)[3,2]^2, tol=1e-3)) and error - check!!!






## TODO: check twoAFC!!
