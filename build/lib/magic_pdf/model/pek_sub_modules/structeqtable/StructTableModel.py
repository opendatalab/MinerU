from struct_eqtable.model import StructTable
from pypandoc import convert_text
class StructTableModel:
    def __init__(self, model_path, max_new_tokens=2048, max_time=400, device = 'cpu'):
        # init
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens # maximum output tokens length
        self.max_time = max_time # timeout for processing in seconds
        if device == 'cuda':
            self.model = StructTable(self.model_path, self.max_new_tokens, self.max_time).cuda()
        else:
            self.model = StructTable(self.model_path, self.max_new_tokens, self.max_time)

    def image2latex(self, image) -> str:
        #
        table_latex = self.model.forward(image)
        return table_latex

    def image2html(self, image) -> str:
        table_latex = self.image2latex(image)
        table_html = convert_text(table_latex, 'html', format='latex')
        return table_html
