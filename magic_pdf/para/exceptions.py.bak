class DenseSingleLineBlockException(Exception):
    """
    This class defines the exception type for dense single line-block.
    """

    def __init__(self, message="DenseSingleLineBlockException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class TitleDetectionException(Exception):
    """
    This class defines the exception type for title detection.
    """

    def __init__(self, message="TitleDetectionException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class TitleLevelException(Exception):
    """
    This class defines the exception type for title level.
    """

    def __init__(self, message="TitleLevelException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class ParaSplitException(Exception):
    """
    This class defines the exception type for paragraph splitting.
    """

    def __init__(self, message="ParaSplitException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class ParaMergeException(Exception):
    """
    This class defines the exception type for paragraph merging.
    """

    def __init__(self, message="ParaMergeException"):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"

    def __repr__(self):
        return f"{self.message}"


class DiscardByException:
    """
    This class discards pdf files by exception
    """

    def __init__(self) -> None:
        pass

    def discard_by_single_line_block(self, pdf_dic, exception: DenseSingleLineBlockException):
        """
        This function discards pdf files by single line block exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        exception_page_nums = 0
        page_num = 0
        for page_id, page in pdf_dic.items():
            if page_id.startswith("page_"):
                page_num += 1
                if "preproc_blocks" in page.keys():
                    preproc_blocks = page["preproc_blocks"]

                    all_single_line_blocks = []
                    for block in preproc_blocks:
                        if len(block["lines"]) == 1:
                            all_single_line_blocks.append(block)

                    if len(preproc_blocks) > 0 and len(all_single_line_blocks) / len(preproc_blocks) > 0.9:
                        exception_page_nums += 1

        if page_num == 0:
            return None

        if exception_page_nums / page_num > 0.1:  # Low ratio means basically, whenever this is the case, it is discarded
            return exception.message

        return None

    def discard_by_title_detection(self, pdf_dic, exception: TitleDetectionException):
        """
        This function discards pdf files by title detection exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None

    def discard_by_title_level(self, pdf_dic, exception: TitleLevelException):
        """
        This function discards pdf files by title level exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None

    def discard_by_split_para(self, pdf_dic, exception: ParaSplitException):
        """
        This function discards pdf files by split para exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None

    def discard_by_merge_para(self, pdf_dic, exception: ParaMergeException):
        """
        This function discards pdf files by merge para exception

        Parameters
        ----------
        pdf_dic : dict
            pdf dictionary
        exception : str
            exception message

        Returns
        -------
        error_message : str
        """
        # return exception.message
        return None
