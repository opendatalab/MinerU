Known Issues
============

-  Reading order is based on the model’s sorting of text distribution in
   space, which may become disordered under extremely complex layouts.
-  Vertical text is not supported.
-  Tables of contents and lists are recognized through rules; a few
   uncommon list formats may not be identified.
-  Only one level of headings is supported; hierarchical heading levels
   are currently not supported.
-  Code blocks are not yet supported in the layout model.
-  Comic books, art books, elementary school textbooks, and exercise
   books are not well-parsed yet
-  Enabling OCR may produce better results in PDFs with a high density
   of formulas
-  If you are processing PDFs with a large number of formulas, it is
   strongly recommended to enable the OCR function. When using PyMuPDF
   to extract text, overlapping text lines can occur, leading to
   inaccurate formula insertion positions.
