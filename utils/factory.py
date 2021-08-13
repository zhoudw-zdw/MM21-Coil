
from models.COIL import COIL
def get_model(model_name, args):
    name = model_name.lower()
    if name=="coil":
        return COIL(args)
    