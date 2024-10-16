import unittest
from PIL import Image
from magic_pdf.model.ppTableModel import ppTableModel

class TestppTableModel(unittest.TestCase):
    def test_image2html(self):
        img = Image.open("tests/test_table/assets/table.jpg")
        # 修改table模型路径
        config = {"device": "cuda",
                  "model_dir": "D:/models/PDF-Extract-Kit/models/TabRec/TableMaster"}
        table_model = ppTableModel(config)
        res = table_model.img2html(img)
        true_value = """<td><table  border="1"><thead><tr><td><b>Methods</b></td><td><b>R</b></td><td><b>P</b></td><td><b>F</b></td><td><b>FPS</b></td></tr></thead><tbody><tr><td>SegLink [26]</td><td>70.0</td><td>86.0</td><td>77.0</td><td>8.9</td></tr><tr><td>PixelLink [4]</td><td>73.2</td><td>83.0</td><td>77.8</td><td>-</td></tr><tr><td>TextSnake [18]</td><td>73.9</td><td>83.2</td><td>78.3</td><td>1.1</td></tr><tr><td>TextField [37]</td><td>75.9</td><td>87.4</td><td>81.3</td><td>5.2 </td></tr><tr><td>MSR[38]</td><td>76.7</td><td>87.4</td><td>81.7</td><td>-</td></tr><tr><td>FTSN[3]</td><td>77.1</td><td>87.6</td><td>82.0</td><td>-</td></tr><tr><td>LSE[30]</td><td>81.7</td><td>84.2</td><td>82.9</td><td>-</td></tr><tr><td>CRAFT [2]</td><td>78.2</td><td>88.2</td><td>82.9</td><td>8.6</td></tr><tr><td>MCN [16]</td><td>79</td><td>88.</td><td>83</td><td>-</td></tr><tr><td>ATRR[35]</td><td>82.1</td><td>85.2</td><td>83.6</td><td>-</td></tr><tr><td>PAN [34]</td><td>83.8</td><td>84.4</td><td>84.1</td><td>30.2</td></tr><tr><td>DB[12]</td><td>79.2</td><td>91.5</td><td>84.9</td><td>32.0</td></tr><tr><td>DRRG [41]</td><td>82.30</td><td>88.05</td><td>85.08</td><td>-</td></tr><tr><td>Ours (SynText)</td><td>80.68</td><td>85.40</td><td>82.97</td><td>12.68</td></tr><tr><td>Ours (MLT-17)</td><td>84.54</td><td>86.62</td><td>85.57</td><td>12.31</td></tr></tbody></table></td>\n"""
        self.assertEqual(true_value, res)


if __name__ == "__main__":
    unittest.main()