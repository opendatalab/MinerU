import json
import numpy as np
import cv2
import math
import re

from PIL import Image, ImageOps
from typing import List, Optional, Tuple, Union, Dict, Any

from loguru import logger
from tokenizers import AddedToken
from tokenizers import Tokenizer as TokenizerFast

from mineru.model.mfr.utils import fix_latex_left_right, fix_latex_environments, remove_up_commands, \
    remove_unsupported_commands


class UniMERNetImgDecode(object):
    """Class for decoding images for UniMERNet, including cropping margins, resizing, and padding."""

    def __init__(
            self, input_size: Tuple[int, int], random_padding: bool = False, **kwargs
    ) -> None:
        """Initializes the UniMERNetImgDecode class with input size and random padding options.

        Args:
            input_size (tuple): The desired input size for the images (height, width).
            random_padding (bool): Whether to use random padding for resizing.
            **kwargs: Additional keyword arguments."""
        self.input_size = input_size
        self.random_padding = random_padding

    def crop_margin(self, img: Image.Image) -> Image.Image:
        """Crops the margin of the image based on grayscale thresholding.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The cropped image."""
        data = np.array(img.convert("L"))
        data = data.astype(np.uint8)
        max_val = data.max()
        min_val = data.min()
        if max_val == min_val:
            return img
        data = (data - min_val) / (max_val - min_val) * 255
        gray = 255 * (data < 200).astype(np.uint8)
        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        return img.crop((a, b, w + a, h + b))

    def get_dimensions(self, img: Union[Image.Image, np.ndarray]) -> List[int]:
        """Gets the dimensions of the image.

        Args:
            img (PIL.Image.Image or numpy.ndarray): The input image.

        Returns:
            list: A list containing the number of channels, height, and width."""
        if hasattr(img, "getbands"):
            channels = len(img.getbands())
        else:
            channels = img.channels
        width, height = img.size
        return [channels, height, width]

    def _compute_resized_output_size(
            self,
            image_size: Tuple[int, int],
            size: Union[int, Tuple[int, int]],
            max_size: Optional[int] = None,
    ) -> List[int]:
        """Computes the resized output size of the image.

        Args:
            image_size (tuple): The original size of the image (height, width).
            size (int or tuple): The desired size for the smallest edge or both height and width.
            max_size (int, optional): The maximum allowed size for the longer edge.

        Returns:
            list: A list containing the new height and width."""
        if len(size) == 1:  # specified size only for the smallest edge
            h, w = image_size
            short, long = (w, h) if w <= h else (h, w)
            requested_new_short = size if isinstance(size, int) else size[0]

            new_short, new_long = requested_new_short, int(
                requested_new_short * long / short
            )

            if max_size is not None:
                if max_size <= requested_new_short:
                    raise ValueError(
                        f"max_size = {max_size} must be strictly greater than the requested "
                        f"size for the smaller edge size = {size}"
                    )
                if new_long > max_size:
                    new_short, new_long = int(max_size * new_short / new_long), max_size

            new_w, new_h = (new_short, new_long) if w <= h else (new_long, new_short)
        else:  # specified both h and w
            new_w, new_h = size[1], size[0]
        return [new_h, new_w]

    def resize(
            self, img: Image.Image, size: Union[int, Tuple[int, int]]
    ) -> Image.Image:
        """Resizes the image to the specified size.

        Args:
            img (PIL.Image.Image): The input image.
            size (int or tuple): The desired size for the smallest edge or both height and width.

        Returns:
            PIL.Image.Image: The resized image."""
        _, image_height, image_width = self.get_dimensions(img)
        if isinstance(size, int):
            size = [size]
        max_size = None
        output_size = self._compute_resized_output_size(
            (image_height, image_width), size, max_size
        )
        img = img.resize(tuple(output_size[::-1]), resample=2)
        return img

    def img_decode(self, img: np.ndarray) -> Optional[np.ndarray]:
        """Decodes the image by cropping margins, resizing, and adding padding.

        Args:
            img (numpy.ndarray): The input image array.

        Returns:
            numpy.ndarray: The decoded image array."""
        try:
            img = self.crop_margin(Image.fromarray(img).convert("RGB"))
        except OSError:
            return
        if img.height == 0 or img.width == 0:
            return
        img = self.resize(img, min(self.input_size))
        img.thumbnail((self.input_size[1], self.input_size[0]))
        delta_width = self.input_size[1] - img.width
        delta_height = self.input_size[0] - img.height
        if self.random_padding:
            pad_width = np.random.randint(low=0, high=delta_width + 1)
            pad_height = np.random.randint(low=0, high=delta_height + 1)
        else:
            pad_width = delta_width // 2
            pad_height = delta_height // 2
        padding = (
            pad_width,
            pad_height,
            delta_width - pad_width,
            delta_height - pad_height,
        )
        return np.array(ImageOps.expand(img, padding))

    def __call__(self, imgs: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Calls the img_decode method on a list of images.

        Args:
            imgs (list of numpy.ndarray): The list of input image arrays.

        Returns:
            list of numpy.ndarray: The list of decoded image arrays."""
        return [self.img_decode(img) for img in imgs]


class UniMERNetTestTransform:
    """
    A class for transforming images according to UniMERNet test specifications.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the UniMERNetTestTransform class.
        """
        super().__init__()
        self.num_output_channels = 3

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Transforms a single image for UniMERNet testing.

        Args:
            img (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The transformed image.
        """
        mean = [0.7931, 0.7931, 0.7931]
        std = [0.1738, 0.1738, 0.1738]
        scale = float(1 / 255.0)
        shape = (1, 1, 3)
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")
        img = (img.astype("float32") * scale - mean) / std
        grayscale_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        squeezed = np.squeeze(grayscale_image)
        img = cv2.merge([squeezed] * self.num_output_channels)
        return img

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """
        Applies the transform to a list of images.

        Args:
            imgs (list of numpy.ndarray): The list of input images.

        Returns:
            list of numpy.ndarray: The list of transformed images.
        """
        return [self.transform(img) for img in imgs]


class LatexImageFormat:
    """Class for formatting images to a specific format suitable for LaTeX."""

    def __init__(self, **kwargs) -> None:
        """Initializes the LatexImageFormat class with optional keyword arguments."""
        super().__init__()

    def format(self, img: np.ndarray) -> np.ndarray:
        """Formats a single image to the LaTeX-compatible format.

        Args:
            img (numpy.ndarray): The input image as a numpy array.

        Returns:
            numpy.ndarray: The formatted image as a numpy array with an added dimension for color.
        """
        im_h, im_w = img.shape[:2]
        divide_h = math.ceil(im_h / 16) * 16
        divide_w = math.ceil(im_w / 16) * 16
        img = img[:, :, 0]
        img = np.pad(
            img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1)
        )
        img_expanded = img[:, :, np.newaxis].transpose(2, 0, 1)
        return img_expanded[np.newaxis, :]

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Applies the format method to a list of images.

        Args:
            imgs (list of numpy.ndarray): A list of input images as numpy arrays.

        Returns:
            list of numpy.ndarray: A list of formatted images as numpy arrays.
        """
        return [self.format(img) for img in imgs]


class ToBatch(object):
    """A class for batching images."""

    def __init__(self, **kwargs) -> None:
        """Initializes the ToBatch object."""
        super(ToBatch, self).__init__()

    def __call__(self, imgs: List[np.ndarray]) -> List[np.ndarray]:
        """Concatenates a list of images into a single batch.

        Args:
            imgs (list): A list of image arrays to be concatenated.

        Returns:
            list: A list containing the concatenated batch of images wrapped in another list (to comply with common batch processing formats).
        """
        batch_imgs = np.concatenate(imgs)
        batch_imgs = batch_imgs.copy()
        x = [batch_imgs]
        return x


class UniMERNetDecode(object):
    """Class for decoding tokenized inputs using UniMERNet tokenizer.

    Attributes:
        SPECIAL_TOKENS_ATTRIBUTES (List[str]): List of special token attributes.
        model_input_names (List[str]): List of model input names.
        max_seq_len (int): Maximum sequence length.
        pad_token_id (int): ID for the padding token.
        bos_token_id (int): ID for the beginning-of-sequence token.
        eos_token_id (int): ID for the end-of-sequence token.
        padding_side (str): Padding side, either 'left' or 'right'.
        pad_token (str): Padding token.
        pad_token_type_id (int): Type ID for the padding token.
        pad_to_multiple_of (Optional[int]): If set, pad to a multiple of this value.
        tokenizer (TokenizerFast): Fast tokenizer instance.

    Args:
        character_list (Dict[str, Any]): Dictionary containing tokenizer configuration.
        **kwargs: Additional keyword arguments.
    """

    SPECIAL_TOKENS_ATTRIBUTES = [
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    ]

    def __init__(
            self,
            character_list: Dict[str, Any],
            **kwargs,
    ) -> None:
        """Initializes the UniMERNetDecode class.

        Args:
            character_list (Dict[str, Any]): Dictionary containing tokenizer configuration.
            **kwargs: Additional keyword arguments.
        """

        self._unk_token = "<unk>"
        self._bos_token = "<s>"
        self._eos_token = "</s>"
        self._pad_token = "<pad>"
        self._sep_token = None
        self._cls_token = None
        self._mask_token = None
        self._additional_special_tokens = []
        self.model_input_names = ["input_ids", "token_type_ids", "attention_mask"]
        self.max_seq_len = 2048
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.padding_side = "right"
        self.pad_token_id = 1
        self.pad_token = "<pad>"
        self.pad_token_type_id = 0
        self.pad_to_multiple_of = None

        fast_tokenizer_str = json.dumps(character_list["fast_tokenizer_file"])
        fast_tokenizer_buffer = fast_tokenizer_str.encode("utf-8")
        self.tokenizer = TokenizerFast.from_buffer(fast_tokenizer_buffer)
        tokenizer_config = (
            character_list["tokenizer_config_file"]
            if "tokenizer_config_file" in character_list
            else None
        )
        added_tokens_decoder = {}
        added_tokens_map = {}
        if tokenizer_config is not None:
            init_kwargs = tokenizer_config
            if "added_tokens_decoder" in init_kwargs:
                for idx, token in init_kwargs["added_tokens_decoder"].items():
                    if isinstance(token, dict):
                        token = AddedToken(**token)
                    if isinstance(token, AddedToken):
                        added_tokens_decoder[int(idx)] = token
                        added_tokens_map[str(token)] = token
                    else:
                        raise ValueError(
                            f"Found a {token.__class__} in the saved `added_tokens_decoder`, should be a dictionary or an AddedToken instance"
                        )
            init_kwargs["added_tokens_decoder"] = added_tokens_decoder
            added_tokens_decoder = init_kwargs.pop("added_tokens_decoder", {})
            tokens_to_add = [
                token
                for index, token in sorted(
                    added_tokens_decoder.items(), key=lambda x: x[0]
                )
                if token not in added_tokens_decoder
            ]
            added_tokens_encoder = self.added_tokens_encoder(added_tokens_decoder)
            encoder = list(added_tokens_encoder.keys()) + [
                str(token) for token in tokens_to_add
            ]
            tokens_to_add += [
                token
                for token in self.all_special_tokens_extended
                if token not in encoder and token not in tokens_to_add
            ]
            if len(tokens_to_add) > 0:
                is_last_special = None
                tokens = []
                special_tokens = self.all_special_tokens
                for token in tokens_to_add:
                    is_special = (
                        (token.special or str(token) in special_tokens)
                        if isinstance(token, AddedToken)
                        else str(token) in special_tokens
                    )
                    if is_last_special is None or is_last_special == is_special:
                        tokens.append(token)
                    else:
                        self._add_tokens(tokens, special_tokens=is_last_special)
                        tokens = [token]
                    is_last_special = is_special
                if tokens:
                    self._add_tokens(tokens, special_tokens=is_last_special)

    def _add_tokens(
            self, new_tokens: "List[Union[AddedToken, str]]", special_tokens: bool = False
    ) -> "List[Union[AddedToken, str]]":
        """Adds new tokens to the tokenizer.

        Args:
            new_tokens (List[Union[AddedToken, str]]): Tokens to be added.
            special_tokens (bool): Indicates whether the tokens are special tokens.

        Returns:
            List[Union[AddedToken, str]]: added tokens.
        """
        if special_tokens:
            return self.tokenizer.add_special_tokens(new_tokens)

        return self.tokenizer.add_tokens(new_tokens)

    def added_tokens_encoder(
            self, added_tokens_decoder: "Dict[int, AddedToken]"
    ) -> Dict[str, int]:
        """Creates an encoder dictionary from added tokens.

        Args:
            added_tokens_decoder (Dict[int, AddedToken]): Dictionary mapping token IDs to tokens.

        Returns:
            Dict[str, int]: Dictionary mapping token strings to IDs.
        """
        return {
            k.content: v
            for v, k in sorted(added_tokens_decoder.items(), key=lambda item: item[0])
        }

    @property
    def all_special_tokens(self) -> List[str]:
        """Retrieves all special tokens.

        Returns:
            List[str]: List of all special tokens as strings.
        """
        all_toks = [str(s) for s in self.all_special_tokens_extended]
        return all_toks

    @property
    def all_special_tokens_extended(self) -> "List[Union[str, AddedToken]]":
        """Retrieves all special tokens, including extended ones.

        Returns:
            List[Union[str, AddedToken]]: List of all special tokens.
        """
        all_tokens = []
        seen = set()
        for value in self.special_tokens_map_extended.values():
            if isinstance(value, (list, tuple)):
                tokens_to_add = [token for token in value if str(token) not in seen]
            else:
                tokens_to_add = [value] if str(value) not in seen else []
            seen.update(map(str, tokens_to_add))
            all_tokens.extend(tokens_to_add)
        return all_tokens

    @property
    def special_tokens_map_extended(self) -> Dict[str, Union[str, List[str]]]:
        """Retrieves the extended map of special tokens.

        Returns:
            Dict[str, Union[str, List[str]]]: Dictionary mapping special token attributes to their values.
        """
        set_attr = {}
        for attr in self.SPECIAL_TOKENS_ATTRIBUTES:
            attr_value = getattr(self, "_" + attr)
            if attr_value:
                set_attr[attr] = attr_value
        return set_attr

    def convert_ids_to_tokens(
            self, ids: Union[int, List[int]], skip_special_tokens: bool = False
    ) -> Union[str, List[str]]:
        """Converts token IDs to token strings.

        Args:
            ids (Union[int, List[int]]): Token ID(s) to convert.
            skip_special_tokens (bool): Whether to skip special tokens during conversion.

        Returns:
            Union[str, List[str]]: Converted token string(s).
        """
        if isinstance(ids, int):
            return self.tokenizer.id_to_token(ids)
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids:
                continue
            tokens.append(self.tokenizer.id_to_token(index))
        return tokens

    def detokenize(self, tokens: List[List[int]]) -> List[List[str]]:
        """Detokenizes a list of token IDs back into strings.

        Args:
            tokens (List[List[int]]): List of token ID lists.

        Returns:
            List[List[str]]: List of detokenized strings.
        """
        self.tokenizer.bos_token = "<s>"
        self.tokenizer.eos_token = "</s>"
        self.tokenizer.pad_token = "<pad>"
        toks = [self.convert_ids_to_tokens(tok) for tok in tokens]
        for b in range(len(toks)):
            for i in reversed(range(len(toks[b]))):
                if toks[b][i] is None:
                    toks[b][i] = ""
                toks[b][i] = toks[b][i].replace("Ġ", " ").strip()
                if toks[b][i] in (
                        [
                            self.tokenizer.bos_token,
                            self.tokenizer.eos_token,
                            self.tokenizer.pad_token,
                        ]
                ):
                    del toks[b][i]
        return toks

    def token2str(self, token_ids: List[List[int]]) -> List[str]:
        """Converts a list of token IDs to strings.

        Args:
            token_ids (List[List[int]]): List of token ID lists.

        Returns:
            List[str]: List of converted strings.
        """
        generated_text = []
        for tok_id in token_ids:
            end_idx = np.argwhere(tok_id == 2)
            if len(end_idx) > 0:
                end_idx = int(end_idx[0][0])
                tok_id = tok_id[: end_idx + 1]
            generated_text.append(
                self.tokenizer.decode(tok_id, skip_special_tokens=True)
            )
        generated_text = [self.post_process(text) for text in generated_text]
        return generated_text

    def normalize(self, s: str) -> str:
        """Normalizes a string by removing unnecessary spaces.

        Args:
            s (str): String to normalize.

        Returns:
            str: Normalized string.
        """
        text_reg = r"(\\(operatorname|mathrm|text|mathbf)\s?\*? {.*?})"
        letter = "[a-zA-Z]"
        noletter = r"[\W_^\d]"
        names = []
        for x in re.findall(text_reg, s):
            pattern = r"(\\[a-zA-Z]+)\s(?=\w)|\\[a-zA-Z]+\s(?=})"
            matches = re.findall(pattern, x[0])
            for m in matches:
                if (
                        m
                        not in [
                    "\\operatorname",
                    "\\mathrm",
                    "\\text",
                    "\\mathbf",
                ]
                        and m.strip() != ""
                ):
                    s = s.replace(m, m + "XXXXXXX")
                    s = s.replace(" ", "")
                    names.append(s)
        if len(names) > 0:
            s = re.sub(text_reg, lambda match: str(names.pop(0)), s)
        news = s
        while True:
            s = news
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, noletter), r"\1\2", s)
            news = re.sub(r"(?!\\ )(%s)\s+?(%s)" % (noletter, letter), r"\1\2", news)
            news = re.sub(r"(%s)\s+?(%s)" % (letter, noletter), r"\1\2", news)
            if news == s:
                break
        return s.replace("XXXXXXX", " ")

    def remove_chinese_text_wrapping(self, formula):
        pattern = re.compile(r"\\text\s*{\s*([^}]*?[\u4e00-\u9fff]+[^}]*?)\s*}")

        def replacer(match):
            return match.group(1)

        replaced_formula = pattern.sub(replacer, formula)
        return replaced_formula.replace('"', "")

    def post_process(self, text: str) -> str:
        """Post-processes a string by fixing text and normalizing it.

        Args:
            text (str): String to post-process.

        Returns:
            str: Post-processed string.
        """
        from ftfy import fix_text

        text = self.remove_chinese_text_wrapping(text)
        text = fix_text(text)
        # logger.debug(f"Text after ftfy fix: {text}")
        text = self.fix_latex(text)
        # logger.debug(f"Text after LaTeX fix: {text}")
        return text

    def fix_latex(self, text: str) -> str:
        """Fixes LaTeX formatting in a string.

        Args:
            text (str): String to fix.

        Returns:
            str: Fixed string.
        """
        text = fix_latex_left_right(text, fix_delimiter=False)
        text = fix_latex_environments(text)
        text = remove_up_commands(text)
        text = remove_unsupported_commands(text)
        # text = self.normalize(text)
        return text

    def __call__(
            self,
            preds: np.ndarray,
            label: Optional[np.ndarray] = None,
            mode: str = "eval",
            *args,
            **kwargs,
    ) -> Union[List[str], tuple]:
        """Processes predictions and optionally labels, returning the decoded text.

        Args:
            preds (np.ndarray): Model predictions.
            label (Optional[np.ndarray]): True labels, if available.
            mode (str): Mode of operation, either 'train' or 'eval'.

        Returns:
            Union[List[str], tuple]: Decoded text, optionally with labels.
        """
        if mode == "train":
            preds_idx = np.array(preds.argmax(axis=2))
            text = self.token2str(preds_idx)
        else:
            text = self.token2str(np.array(preds))
        if label is None:
            return text
        label = self.token2str(np.array(label))
        return text, label
