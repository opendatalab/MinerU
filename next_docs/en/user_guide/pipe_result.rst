

Pipe Result
==============

.. admonition:: Tip
    :class: tip

    Please first navigate to :doc:`tutorial/pipeline` to get an initial understanding of how the pipeline works; this will help in understanding the content of this section.


The **PipeResult** class is a container for storing pipeline processing results and implements a series of methods related to these results, such as draw_layout, draw_span.
Checkout :doc:`../api/pipe_operators` for more details about **PipeResult**



Structure Definitions
-------------------------------

**some_pdf_middle.json**

+----------------+--------------------------------------------------------------+
| Field Name     | Description                                                  |
|                |                                                              |
+================+==============================================================+
| pdf_info       | list, each element is a dict representing the parsing result |
|                | of each PDF page, see the table below for details            |
+----------------+--------------------------------------------------------------+
| \_             | ocr \| txt, used to indicate the mode used in this           |
| parse_type     | intermediate parsing state                                   |
|                |                                                              |
+----------------+--------------------------------------------------------------+
| \_version_name | string, indicates the version of magic-pdf used in this      |
|                | parsing                                                      |
|                |                                                              |
+----------------+--------------------------------------------------------------+

**pdf_info**

Field structure description

+-------------------------+------------------------------------------------------------+
| Field                   | Description                                                |
| Name                    |                                                            |
+=========================+============================================================+
| preproc_blocks          | Intermediate result after PDF preprocessing, not yet       |
|                         | segmented                                                  |
+-------------------------+------------------------------------------------------------+
| layout_bboxes           | Layout segmentation results, containing layout direction   |
|                         | (vertical, horizontal), and bbox, sorted by reading order  |
+-------------------------+------------------------------------------------------------+
| page_idx                | Page number, starting from 0                               |
|                         |                                                            |
+-------------------------+------------------------------------------------------------+
| page_size               | Page width and height                                      |
|                         |                                                            |
+-------------------------+------------------------------------------------------------+
| \_layout_tree           | Layout tree structure                                      |
|                         |                                                            |
+-------------------------+------------------------------------------------------------+
| images                  | list, each element is a dict representing an img_block     |
+-------------------------+------------------------------------------------------------+
| tables                  | list, each element is a dict representing a table_block    |
+-------------------------+------------------------------------------------------------+
| interline_equation      | list, each element is a dict representing an               |
|                         | interline_equation_block                                   |
|                         |                                                            |
+-------------------------+------------------------------------------------------------+
| discarded_blocks        | List, block information returned by the model that needs   |
|                         | to be dropped                                              |
|                         |                                                            |
+-------------------------+------------------------------------------------------------+
| para_blocks             | Result after segmenting preproc_blocks                     |
|                         |                                                            |
+-------------------------+------------------------------------------------------------+

In the above table, ``para_blocks`` is an array of dicts, each dict
representing a block structure. A block can support up to one level of
nesting.

**block**

The outer block is referred to as a first-level block, and the fields in
the first-level block include:

+------------------------+-------------------------------------------------------------+
| Field                  | Description                                                 |
| Name                   |                                                             |
+========================+=============================================================+
| type                   | Block type (table|image)                                    |
+------------------------+-------------------------------------------------------------+
| bbox                   | Block bounding box coordinates                              |
+------------------------+-------------------------------------------------------------+
| blocks                 | list, each element is a dict representing a second-level    |
|                        | block                                                       |
+------------------------+-------------------------------------------------------------+

There are only two types of first-level blocks: “table” and “image”. All
other blocks are second-level blocks.

The fields in a second-level block include:

+----------------------+----------------------------------------------------------------+
| Field                | Description                                                    |
| Name                 |                                                                |
+======================+================================================================+
|                      | Block type                                                     |
| type                 |                                                                |
+----------------------+----------------------------------------------------------------+
|                      | Block bounding box coordinates                                 |
| bbox                 |                                                                |
+----------------------+----------------------------------------------------------------+
|                      | list, each element is a dict representing a line, used to      |
| lines                | describe the composition of a line of information              |
+----------------------+----------------------------------------------------------------+

Detailed explanation of second-level block types

================== ======================
type               Description
================== ======================
image_body         Main body of the image
image_caption      Image description text
table_body         Main body of the table
table_caption      Table description text
table_footnote     Table footnote
text               Text block
title              Title block
interline_equation Block formula
================== ======================

**line**

The field format of a line is as follows:

+---------------------+----------------------------------------------------------------+
| Field               | Description                                                    |
| Name                |                                                                |
+=====================+================================================================+
|                     | Bounding box coordinates of the line                           |
| bbox                |                                                                |
+---------------------+----------------------------------------------------------------+
| spans               | list, each element is a dict representing a span, used to      |
|                     | describe the composition of the smallest unit                  |
+---------------------+----------------------------------------------------------------+

**span**

+---------------------+-----------------------------------------------------------+
| Field               | Description                                               |
| Name                |                                                           |
+=====================+===========================================================+
| bbox                | Bounding box coordinates of the span                      |
+---------------------+-----------------------------------------------------------+
| type                | Type of the span                                          |
+---------------------+-----------------------------------------------------------+
| content             | Text spans use content, chart spans use img_path to store |
| \|                  | the actual text or screenshot path information            |
| img_path            |                                                           |
+---------------------+-----------------------------------------------------------+

The types of spans are as follows:

================== ==============
type               Description
================== ==============
image              Image
table              Table
text               Text
inline_equation    Inline formula
interline_equation Block formula
================== ==============

**Summary**

A span is the smallest storage unit for all elements.

The elements stored within para_blocks are block information.

The block structure is as follows:

First-level block (if any) -> Second-level block -> Line -> Span

.. _example-1:

example
^^^^^^^

.. code:: json

   {
       "pdf_info": [
           {
               "preproc_blocks": [
                   {
                       "type": "text",
                       "bbox": [
                           52,
                           61.956024169921875,
                           294,
                           82.99800872802734
                       ],
                       "lines": [
                           {
                               "bbox": [
                                   52,
                                   61.956024169921875,
                                   294,
                                   72.0000228881836
                               ],
                               "spans": [
                                   {
                                       "bbox": [
                                           54.0,
                                           61.956024169921875,
                                           296.2261657714844,
                                           72.0000228881836
                                       ],
                                       "content": "dependent on the service headway and the reliability of the departure ",
                                       "type": "text",
                                       "score": 1.0
                                   }
                               ]
                           }
                       ]
                   }
               ],
               "layout_bboxes": [
                   {
                       "layout_bbox": [
                           52,
                           61,
                           294,
                           731
                       ],
                       "layout_label": "V",
                       "sub_layout": []
                   }
               ],
               "page_idx": 0,
               "page_size": [
                   612.0,
                   792.0
               ],
               "_layout_tree": [],
               "images": [],
               "tables": [],
               "interline_equations": [],
               "discarded_blocks": [],
               "para_blocks": [
                   {
                       "type": "text",
                       "bbox": [
                           52,
                           61.956024169921875,
                           294,
                           82.99800872802734
                       ],
                       "lines": [
                           {
                               "bbox": [
                                   52,
                                   61.956024169921875,
                                   294,
                                   72.0000228881836
                               ],
                               "spans": [
                                   {
                                       "bbox": [
                                           54.0,
                                           61.956024169921875,
                                           296.2261657714844,
                                           72.0000228881836
                                       ],
                                       "content": "dependent on the service headway and the reliability of the departure ",
                                       "type": "text",
                                       "score": 1.0
                                   }
                               ]
                           }
                       ]
                   }
               ]
           }
       ],
       "_parse_type": "txt",
       "_version_name": "0.6.1"
   }


Pipeline Result
------------------

.. code:: python

    from magic_pdf.pdf_parse_union_core_v2 import pdf_parse_union
    from magic_pdf.operators.pipes import PipeResult
    from magic_pdf.data.dataset import Dataset

    res = pdf_parse_union(*args, **kwargs)
    res['_parse_type'] = PARSE_TYPE_OCR
    res['_version_name'] = __version__
    if 'lang' in kwargs and kwargs['lang'] is not None:
        res['lang'] = kwargs['lang']

    dataset : Dataset = some_dataset   # not real dataset
    pipeResult = PipeResult(res, dataset)



some_pdf_layout.pdf
~~~~~~~~~~~~~~~~~~~

Each page layout consists of one or more boxes. The number at the top
left of each box indicates its sequence number. Additionally, in
``layout.pdf``, different content blocks are highlighted with different
background colors.

.. figure:: ../_static/image/layout_example.png
   :alt: layout example

   layout example

some_pdf_spans.pdf
~~~~~~~~~~~~~~~~~~

All spans on the page are drawn with different colored line frames
according to the span type. This file can be used for quality control,
allowing for quick identification of issues such as missing text or
unrecognized inline formulas.

.. figure:: ../_static/image/spans_example.png
   :alt: spans example

   spans example
