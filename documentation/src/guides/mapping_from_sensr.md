---
layout: layouts/base.njk
title: "Mapping from sensR to sensPy"
permalink: /guides/mapping-from-sensr/
tags: guide
---

## Introduction

This page provides a mapping to help users familiar with the R package `sensR` find equivalent functionalities within the `sensPy` Python library. While `sensPy` aims to offer similar capabilities, function names, parameters, and output structures may differ to align with Python conventions and the library's specific design.

## Function Mapping Table

Below is a table mapping functions from the original `sensR` package to their equivalents in `sensPy`.

<table>
  <thead>
    <tr>
      <th><code>sensR</code> Function</th>
      <th><code>sensR</code> File</th>
      <th><code>sensPy</code> Equivalent</th>
      <th><code>sensPy</code> Module</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
  {%- for item in sensr_senspy_mapping -%}
    <tr>
      <td><code>{{ item.sensR_function }}</code></td>
      <td>{{ item.sensR_file }}</td>
      <td><code>{{ item.senspy_function }}</code></td>
      <td><code>{{ item.senspy_module }}</code></td>
      <td>{{ item.description }}</td>
    </tr>
  {%- endfor -%}
  </tbody>
</table>
