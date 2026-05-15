import sys

class CustomException(Exception):
    def __init__(self, message, error_detail: Exception= None):
        self.error_message = self.get_detailed_error_message(message, error_detail)
        super().__init__(self.error_message)

    @staticmethod
    def get_detailed_error_message(message, error_detail: Exception):
        exc_type, exc_value, exc_traceback = error_detail.exc_info()
        file_name = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        
        return f"Error: {message}\nError detail: {exc_value}\nFile name: {file_name}\nLine number: {line_number}"

    def __str__(self):
        return self.error_message