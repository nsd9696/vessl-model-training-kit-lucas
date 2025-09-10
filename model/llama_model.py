from llama_cpp import Llama
from typing import Any, Dict, List
import time
MAX_GENERATION_LENGTH = 512

class LlamaCppModelRunner:
    def __init__(self, model_path: str, **kwargs):
        self.model = Llama(
            model_path=model_path,
            logits_all=True,
            verbose=False,
            n_gpu_layers=-1,
            n_ctx=16384,
            **kwargs
            
        )

        self.get_target_tokens()
    def get_target_tokens(self):
        ## Get target_tokens 
        labels = [b"a",b"b",b"c",b"d",b"e"]
        self.target_tokens = [self.model.tokenize(text = text)[0] for text in labels]
        self.logit_bias = {i:-10000 for i in range(256*256) if i not in self.target_tokens}
    def call_completion(self,prompt, is_thinking: bool = False):
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]  
        if is_thinking:
            MAX_GENERATION_LENGTH = 4096
        return self.model.create_chat_completion(
                messages=prompt,
                response_format=None,
                temperature=0.0,
                max_tokens=MAX_GENERATION_LENGTH,
            )
        
    def predict_generation(self,prompts, is_thinking: bool = False):
        results = []
        start_time = time.time()
        for prompt in prompts:
            
            results.append(self.call_completion(prompt, is_thinking))
        end_time = time.time()
        print(f"Generation time: {end_time - start_time} seconds")
        contents = [result['choices'][0]['message']['content'] for result in results]
        if is_thinking:
            contents = [content.split("</think>")[-1] for content in contents]
        return {"responses": contents}

def load_llama_model_runner(model_name: str):
    model_runner = LlamaCppModelRunner(model_name)
    return model_runner