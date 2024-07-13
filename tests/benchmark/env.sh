conda create -n MinerU python=3.10
conda activate MinerU
pip install magic-pdf
pip install magic-pdf[full-cpu]
pip install detectron2 --extra-index-url https://myhloli.github.io/wheels/
git lfs install
git lfs clone https://huggingface.co/wanderkid/PDF-Extract-Kit
#cp magic-pdf.template.json ~/magic-pdf.json