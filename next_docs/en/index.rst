.. xtuner documentation master file, created by
   sphinx-quickstart on Tue Jan  9 16:33:06 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the MinerU Documentation
==============================================

.. figure:: ./_static/image/logo.png
  :align: center
  :alt: mineru
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>A one-stop, open-source, high-quality data extraction tool
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/opendatalab/MinerU" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/opendatalab/MinerU/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/opendatalab/MinerU/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>


Project Introduction
--------------------

MinerU is a tool that converts PDFs into machine-readable formats (e.g.,
markdown, JSON), allowing for easy extraction into any format. MinerU
was born during the pre-training process of
`InternLM <https://github.com/InternLM/InternLM>`__. We focus on solving
symbol conversion issues in scientific literature and hope to contribute
to technological development in the era of large models. Compared to
well-known commercial products, MinerU is still young. If you encounter
any issues or if the results are not as expected, please submit an issue
on `issue <https://github.com/opendatalab/MinerU/issues>`__ and **attach
the relevant PDF**.

.. video:: https://github.com/user-attachments/assets/4bea02c9-6d54-4cd6-97ed-dff14340982c


Key Features
------------

-  Removes elements such as headers, footers, footnotes, and page
   numbers while maintaining semantic continuity
-  Outputs text in a human-readable order from multi-column documents
-  Retains the original structure of the document, including titles,
   paragraphs, and lists
-  Extracts images, image captions, tables, and table captions
-  Automatically recognizes formulas in the document and converts them
   to LaTeX
-  Automatically recognizes tables in the document and converts them to
   LaTeX
-  Automatically detects and enables OCR for corrupted PDFs
-  Supports both CPU and GPU environments
-  Supports Windows, Linux, and Mac platforms


User Guide
-------------
.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide


API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
   :maxdepth: 2
   :caption: API

   api


Additional Notes
------------------
.. toctree::
   :maxdepth: 1
   :caption: Additional Notes

   additional_notes/known_issues
   additional_notes/faq
   additional_notes/changelog
   additional_notes/glossary


Projects 
---------
.. toctree::
   :maxdepth: 1
   :caption: Projects

   projects