<div align="center" xmlns="http://www.w3.org/1999/html">
<!-- logo -->
<p align="center">
  <img src="docs/images/MinerU-logo.png" width="300px" style="vertical-align:middle;">
</p>

<!-- icon -->
[![stars](https://img.shields.io/github/stars/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![forks](https://img.shields.io/github/forks/opendatalab/MinerU.svg)](https://github.com/opendatalab/MinerU)
[![open issues](https://img.shields.io/github/issues-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/opendatalab/MinerU)](https://github.com/opendatalab/MinerU/issues)
[![PyPI version](https://badge.fury.io/py/magic-pdf.svg)](https://badge.fury.io/py/magic-pdf)
[![Downloads](https://static.pepy.tech/badge/magic-pdf)](https://pepy.tech/project/magic-pdf)
[![Downloads](https://static.pepy.tech/badge/magic-pdf/month)](https://pepy.tech/project/magic-pdf)
<a href="https://trendshift.io/repositories/11174" target="_blank"><img src="https://trendshift.io/api/badge/repositories/11174" alt="opendatalab%2FMinerU | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

<!-- language -->
[English](README.md) | [简体中文](README_zh-CN.md)

<!-- hot link -->
<p align="center">
<a href="https://github.com/opendatalab/PDF-Extract-Kit">PDF-Extract-Kit: High-Quality PDF Extraction Toolkit</a>🔥🔥🔥
</p>

<!-- join us -->
<p align="center">
    👋 join us on <a href="https://discord.gg/gPxmVeGC" target="_blank">Discord</a> and <a href="https://cdn.vansin.top/internlm/mineru.jpg" target="_blank">WeChat</a>
</p>

</div>

# Changelog
- 2024/08/09: Version 0.7.0b1 released, simplified installation process, added table recognition functionality
- 2024/08/01: Version 0.6.2b1 released, optimized dependency conflict issues and installation documentation
- 2024/07/05: Initial open-source release

<!-- TABLE OF CONTENT -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#mineru">MinerU</a>
      <ul>
        <li><a href="#project-introduction">Project Introduction</a></li>
        <li><a href="#key-features">Key Features</a></li>
        <li><a href="#quick-start">Quick Start</a>
            <ul>
            <li><a href="#online-demo">Online Demo</a></li>
            <li><a href="#quick-cpu-demo">Quick CPU Demo</a></li>
            <li><a href="#gpu-usage">GPU Usage</a></li>
            </ul>
        </li>
        <li><a href="#usage">Usage</a>
            <ul>
            <li><a href="#command-line">Command Line</a></li>
            <li><a href="#api">API</a></li>
            <li><a href="#advanced-development">Advanced Development</a></li>
            </ul>
        </li>
      </ul>
    </li>
    <li><a href="#todo">TODO List</a></li>
    <li><a href="#known-issue">Known Issue</a></li>
    <li><a href="#faq">FAQ</a></li>
    <li><a href="#all-thanks-to-our-contributors">Contributors</a></li>
    <li><a href="#license-information">License Information</a></li>
    <li><a href="#acknowledgments">Acknowledgements</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#star-history">Star History</a></li>
    <li><a href="#magic-doc">Magic-doc: Fast Extraction from PPT/DOC/PDF</a></li>
    <li><a href="#magic-html">Magic-html: Mixed Web Page Extraction</a></li>
    <li><a href="#links">Links</a></li>
  </ol>
</details>

# MinerU
## Project Introduction
MinerU is a tool that converts PDFs into machine-readable formats (e.g., markdown, JSON), allowing for easy extraction into any format.
MinerU was born during the pre-training process of [InternLM](https://github.com/InternLM/InternLM). We focus on solving symbol conversion issues in scientific literature and hope to contribute to technological development in the era of large models.
Compared to well-known commercial products, MinerU is still young. If you encounter any issues or if the results are not as expected, please submit an issue on [issue](https://github.com/opendatalab/MinerU/issues) and **attach the relevant PDF**.

https://github.com/user-attachments/assets/4bea02c9-6d54-4cd6-97ed-dff14340982c

## Key Features

- Removes elements such as headers, footers, footnotes, and page numbers while maintaining semantic continuity
- Outputs text in a human-readable order from multi-column documents
- Retains the original structure of the document, including titles, paragraphs, and lists
- Extracts images, image captions, tables, and table captions
- Automatically recognizes formulas in the document and converts them to LaTeX
- Automatically recognizes tables in the document and converts them to LaTeX
- Automatically detects and enables OCR for corrupted PDFs
- Supports both CPU and GPU environments
- Supports Windows, Linux, and Mac platforms

## Quick Start

If you encounter any installation issues, please first consult the <a href="#faq">FAQ</a>. </br>
If the parsing results are not as expected, refer to the <a href="#known-issue">Known Issues</a>. </br>
There are three different ways to experience MinerU:
- [Online Demo (No Installation Required)](#online-demo)
- [Quick CPU Demo (Windows, Linux, Mac)](#quick-cpu-demo)
- [Linux/Windows + CUDA](#gpu-usage)

**⚠️ Pre-installation Notice—Hardware and Software Environment Support**

To ensure the stability and reliability of the project, we only optimize and test for specific hardware and software environments during development. This ensures that users deploying and running the project on recommended system configurations will get the best performance with the fewest compatibility issues.

By focusing resources on the mainline environment, our team can more efficiently resolve potential bugs and develop new features.

In non-mainline environments, due to the diversity of hardware and software configurations, as well as third-party dependency compatibility issues, we cannot guarantee 100% project availability. Therefore, for users who wish to use this project in non-recommended environments, we suggest carefully reading the documentation and FAQ first. M
