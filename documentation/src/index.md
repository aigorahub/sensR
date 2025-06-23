---
layout: layouts/base.njk
title: Welcome to sensPy
---

## Introduction

`sensPy` is a Python library for sensory discrimination analysis, providing tools and models ported from the R package `sensR`. This documentation will guide you through its functionalities.

### Key Features:
*   Psychometric function utilities
*   Discrimination models (d-prime estimation, 2AC, DoD, Same-Different)
*   Beta-Binomial model for overdispersed data
*   Statistical tests for d-prime values

Browse the modules in the sidebar to learn more.

---

## New to sensPy?

Start with our [Getting Started Tutorial](/tutorials/getting-started/) to see a basic workflow and learn how to perform common sensory data analysis tasks.

## Coming from R's sensR package?

If you're familiar with the `sensR` package in R, our [Mapping from sensR Guide](/guides/mapping-from-sensr/) will help you quickly find the equivalent functions and understand any differences in `sensPy`.

## Explore the API

Dive into the details of specific modules and functions:

*   **Models**: Learn about models like the [BetaBinomial model for overdispersed data](/models/beta-binomial/).
*   **Discrimination Utilities**: Explore functions for various sensory discrimination tasks:
    *   General d-prime estimation: [`discrim`](/discrimination/discrim/)
    *   2-Alternative Choice (Yes/No) tasks: [`twoAC`](/discrimination/twoAC/)
    *   Degree of Difference: [`dod`](/discrimination/dod/)
    *   Same-Different tasks: [`samediff`](/discrimination/samediff/)
    *   Hypothesis tests for d-prime: [`dprime_test` & `dprime_compare`](/discrimination/dprime-tests/)
    *   ROC curve related utilities: [`SDT` & `AUC`](/discrimination/roc-utilities/)
*   **Link Utilities**: Understand functions for converting between d-prime, Pc, and Pd: [`psyfun`, `psyinv`, `psyderiv`, `rescale`](/links/utilities/).
