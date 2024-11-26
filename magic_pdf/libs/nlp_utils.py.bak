import re
from os import path

from collections import Counter

from loguru import logger

# from langdetect import detect
import spacy
import en_core_web_sm
import zh_core_web_sm

from magic_pdf.libs.language import detect_lang


class NLPModels:
    """
    How to upload local models to s3:
        - config aws cli:
            doc\SETUP-CLI.md
            doc\setup_cli.sh
            app\config\__init__.py
        - $ cd {local_dir_storing_models}
        - $ ls models
            en_core_web_sm-3.7.1/
            zh_core_web_sm-3.7.0/
        - $ aws s3 sync models/ s3://llm-infra/models --profile=p_project_norm
        - $ aws s3 --profile=p_project_norm ls  s3://llm-infra/models/
            PRE en_core_web_sm-3.7.1/
            PRE zh_core_web_sm-3.7.0/
    """

    def __init__(self):
        # if OS is windows, set "TMP_DIR" to "D:/tmp"

        home_dir = path.expanduser("~")
        self.default_local_path = path.join(home_dir, ".nlp_models")
        self.default_shared_path = "/share/pdf_processor/nlp_models"
        self.default_hdfs_path = "hdfs://pdf_processor/nlp_models"
        self.default_s3_path = "s3://llm-infra/models"
        self.nlp_models = self.nlp_models = {
            "en_core_web_sm": {
                "type": "spacy",
                "version": "3.7.1",
            },
            "en_core_web_md": {
                "type": "spacy",
                "version": "3.7.1",
            },
            "en_core_web_lg": {
                "type": "spacy",
                "version": "3.7.1",
            },
            "zh_core_web_sm": {
                "type": "spacy",
                "version": "3.7.0",
            },
            "zh_core_web_md": {
                "type": "spacy",
                "version": "3.7.0",
            },
            "zh_core_web_lg": {
                "type": "spacy",
                "version": "3.7.0",
            },
        }
        self.en_core_web_sm_model = en_core_web_sm.load()
        self.zh_core_web_sm_model = zh_core_web_sm.load()

    def load_model(self, model_name, model_type, model_version):
        if (
            model_name in self.nlp_models
            and self.nlp_models[model_name]["type"] == model_type
            and self.nlp_models[model_name]["version"] == model_version
        ):
            return spacy.load(model_name) if spacy.util.is_package(model_name) else None

        else:
            logger.error(f"Unsupported model name or version: {model_name} {model_version}")
            return None

    def detect_language(self, text, use_langdetect=False):
        if len(text) == 0:
            return None
        if use_langdetect:
            # print("use_langdetect")
            # print(detect_lang(text))
            # return detect_lang(text)
            if detect_lang(text) == "zh":
                return "zh"
            else:
                return "en"

        if not use_langdetect:
            en_count = len(re.findall(r"[a-zA-Z]", text))
            cn_count = len(re.findall(r"[\u4e00-\u9fff]", text))

            if en_count > cn_count:
                return "en"

            if cn_count > en_count:
                return "zh"

    def detect_entity_catgr_using_nlp(self, text, threshold=0.5):
        """
        Detect entity categories using NLP models and return the most frequent entity types.

        Parameters
        ----------
        text : str
            Text to be processed.

        Returns
        -------
        str
            The most frequent entity type.
        """
        lang = self.detect_language(text, use_langdetect=True)

        if lang == "en":
            nlp_model = self.en_core_web_sm_model
        elif lang == "zh":
            nlp_model = self.zh_core_web_sm_model
        else:
            # logger.error(f"Unsupported language: {lang}")
            return {}

        # Splitting text into smaller parts
        text_parts = re.split(r"[,;，；、\s & |]+", text)

        text_parts = [part for part in text_parts if not re.match(r"[\d\W]+", part)]  # Remove non-words
        text_combined = " ".join(text_parts)

        try:
            doc = nlp_model(text_combined)
            entity_counts = Counter([ent.label_ for ent in doc.ents])
            word_counts_in_entities = Counter()

            for ent in doc.ents:
                word_counts_in_entities[ent.label_] += len(ent.text.split())

            total_words_in_entities = sum(word_counts_in_entities.values())
            total_words = len([token for token in doc if not token.is_punct])

            if total_words_in_entities == 0 or total_words == 0:
                return None

            entity_percentage = total_words_in_entities / total_words
            if entity_percentage < 0.5:
                return None

            most_common_entity, word_count = word_counts_in_entities.most_common(1)[0]
            entity_percentage = word_count / total_words_in_entities

            if entity_percentage >= threshold:
                return most_common_entity
            else:
                return None
        except Exception as e:
            logger.error(f"Error in entity detection: {e}")
            return None


def __main__():
    nlpModel = NLPModels()

    test_strings = [
        "张三",
        "张三, 李四，王五; 赵六",
        "John Doe",
        "Jane Smith",
        "Lee, John",
        "John Doe, Jane Smith; Alice Johnson，Bob Lee",
        "孙七, Michael Jordan；赵八",
        "David Smith  Michael O'Connor; Kevin ßáçøñ",
        "李雷·韩梅梅, 张三·李四",
        "Charles Robert Darwin, Isaac Newton",
        "莱昂纳多·迪卡普里奥, 杰克·吉伦哈尔",
        "John Doe, Jane Smith; Alice Johnson",
        "张三, 李四，王五; 赵六",
        "Lei Wang, Jia Li, and Xiaojun Chen, LINKE YANG OU, and YUAN ZHANG",
        "Rachel Mills  &  William Barry  &  Susanne B. Haga",
        "Claire Chabut* and Jean-François Bussières",
        "1 Department of Chemistry, Northeastern University, Shenyang 110004, China 2 State Key Laboratory of Polymer Physics and Chemistry, Changchun Institute of Applied Chemistry, Chinese Academy of Sciences, Changchun 130022, China",
        "Changchun",
        "china",
        "Rongjun Song, 1,2 Baoyan Zhang, 1 Baotong Huang, 2 Tao Tang 2",
        "Synergistic Effect of Supported Nickel Catalyst with Intumescent Flame-Retardants on Flame Retardancy and Thermal Stability of Polypropylene",
        "Synergistic Effect of Supported Nickel Catalyst with",
        "Intumescent Flame-Retardants on Flame Retardancy",
        "and Thermal Stability of Polypropylene",
    ]

    for test in test_strings:
        print()
        print(f"Original String: {test}")

        result = nlpModel.detect_entity_catgr_using_nlp(test)
        print(f"Detected entities: {result}")


if __name__ == "__main__":
    __main__()
