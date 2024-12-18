
Inference Result
==================

.. admonition:: Tip
    :class: tip

    Please first navigate to :doc:`tutorial/pipeline` to get an initial understanding of how the pipeline works; this will help in understanding the content of this section.

The **InferenceResult** class is a container for storing model inference results and implements a series of methods related to these results, such as draw_model, dump_model.
Checkout :doc:`../api/model_operators` for more details about **InferenceResult**


Model Inference Result
-----------------------

Structure Definition
^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: python

    from pydantic import BaseModel, Field
    from enum import IntEnum

    class CategoryType(IntEnum):
            title = 0               # Title
            plain_text = 1          # Text
            abandon = 2             # Includes headers, footers, page numbers, and page annotations
            figure = 3              # Image
            figure_caption = 4      # Image description
            table = 5               # Table
            table_caption = 6       # Table description
            table_footnote = 7      # Table footnote
            isolate_formula = 8     # Block formula
            formula_caption = 9     # Formula label

            embedding = 13          # Inline formula
            isolated = 14           # Block formula
            text = 15               # OCR recognition result


    class PageInfo(BaseModel):
        page_no: int = Field(description="Page number, the first page is 0", ge=0)
        height: int = Field(description="Page height", gt=0)
        width: int = Field(description="Page width", ge=0)

    class ObjectInferenceResult(BaseModel):
        category_id: CategoryType = Field(description="Category", ge=0)
        poly: list[float] = Field(description="Quadrilateral coordinates, representing the coordinates of the top-left, top-right, bottom-right, and bottom-left points respectively")
        score: float = Field(description="Confidence of the inference result")
        latex: str | None = Field(description="LaTeX parsing result", default=None)
        html: str | None = Field(description="HTML parsing result", default=None)

    class PageInferenceResults(BaseModel):
            layout_dets: list[ObjectInferenceResult] = Field(description="Page recognition results", ge=0)
            page_info: PageInfo = Field(description="Page metadata")


Example
^^^^^^^^^^^

.. code:: json

    [
        {
            "layout_dets": [
                {
                    "category_id": 2,
                    "poly": [
                        99.1906967163086,
                        100.3119125366211,
                        730.3707885742188,
                        100.3119125366211,
                        730.3707885742188,
                        245.81326293945312,
                        99.1906967163086,
                        245.81326293945312
                    ],
                    "score": 0.9999997615814209
                }
            ],
            "page_info": {
                "page_no": 0,
                "height": 2339,
                "width": 1654
            }
        },
        {
            "layout_dets": [
                {
                    "category_id": 5,
                    "poly": [
                        99.13092803955078,
                        2210.680419921875,
                        497.3183898925781,
                        2210.680419921875,
                        497.3183898925781,
                        2264.78076171875,
                        99.13092803955078,
                        2264.78076171875
                    ],
                    "score": 0.9999997019767761
                }
            ],
            "page_info": {
                "page_no": 1,
                "height": 2339,
                "width": 1654
            }
        }
    ]

The format of the poly coordinates is [x0, y0, x1, y1, x2, y2, x3, y3],
representing the coordinates of the top-left, top-right, bottom-right,
and bottom-left points respectively. |Poly Coordinate Diagram|



Inference Result
-------------------------


.. code:: python

    from magic_pdf.operators.models import InferenceResult
    from magic_pdf.data.dataset import Dataset

    dataset : Dataset = some_data_set    # not real dataset

    # The inference results of all pages, ordered by page number, are stored in a list as the inference results of MinerU
    model_inference_result: list[PageInferenceResults] = []

    Inference_result = InferenceResult(model_inference_result, dataset)



some_model.pdf
^^^^^^^^^^^^^^^^^^^^

.. figure:: ../_static/image/inference_result.png



.. |Poly Coordinate Diagram| image:: ../_static/image/poly.png
