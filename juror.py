

class Juror():
    def __init__(self,name, iv_dict):
        self.name = name
        for iv_name, value in iv_dict.items():
            setattr(self, iv_name, value)
