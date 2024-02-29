import os

class UtilsForTestPara:
    def __init__(self):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(curr_dir)
        assets_dir = os.path.join(parent_dir, "assets")
        self.default_pre_proc_out_dir = os.path.join(assets_dir, "pre_proc_results")

        if not os.path.exists(assets_dir):
            raise FileNotFoundError("The assets directory does not exist. Please check the path.")

    def read_preproc_out_jfiles(self, input_dir=None):
        """
        Read all preproc_out.json files under the directory input_dir

        Parameters
        ----------
        input_dir : str
            The directory where the preproc_out.json files are located.
            The default is default_pre_proc_out_dir.

        Returns
        -------
        preproc_out_jsons : list
            A list of paths of preproc_out.json files.

        """
        if input_dir is None:
            input_dir = self.default_pre_proc_out_dir

        preproc_out_jsons = []
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith("preproc_out.json"):
                    preproc_out_json_abs_path = os.path.join(root, file)
                    preproc_out_jsons.append(preproc_out_json_abs_path)

        return preproc_out_jsons

if __name__ == "__main__":
    utils = UtilsForTestPara()
    preproc_out_jsons = utils.read_preproc_out_jfiles()
    for preproc_out_json in preproc_out_jsons:
        print(preproc_out_json)