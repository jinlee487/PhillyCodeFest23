class PrintWrapper:
    def __init__(self, app):
        self.app = app
    def print(self, str):
        self.app.logger.info(str)
    
    