from utils.path import get_assets_dir
import os

class ModelType():
    def __init__(self, model_name: str):
        openai_models = os.path.join(get_assets_dir(), 'openai_model.txt')
        gemini_models = os.path.join(get_assets_dir(), 'gemini_model.txt')
        with open(openai_models, 'r') as f:
            openai_models = f.read().splitlines()
        with open(gemini_models, 'r') as f:
            gemini_models = f.read().splitlines()
        if model_name in openai_models or model_name in gemini_models:
            self.is_api = True
            if model_name in openai_models:
                self.model_type = "openai"
            elif model_name in gemini_models:
                self.model_type = "gemini"
        else:
            self.is_api = False
            self.model_type = "HF"
    def __call__(self):
        return self.is_api, self.model_type