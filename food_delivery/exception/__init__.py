import sys


def error_message_detail(error, error_detail: sys):
    """
    It returns the error message with the file name, try block line number, exception block line number
    and the error message

    :param error: The error message that was raised
    :param error_detail: sys
    :type error_detail: sys
    :return: The error message
    """

    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    try_block_line_no = exc_tb.tb_lineno
    Exception_block_line_no = exc_tb.tb_frame.f_lineno
    error_message = f"""Python Script :
    [{file_name}] 
    at try block line number : [{try_block_line_no}] and exception block line no : [{Exception_block_line_no}] 
    error message : 
    [{str(error)}]
    """
    return error_message


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        """
        A constructor function that initializes the class.

        :param error_message: The error message that will be displayed to the user
        :param error_detail: This is the error message that you want to display
        :type error_detail: sys
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message

    def __repr__(self) -> str:
        return CustomException.__name__.str()
