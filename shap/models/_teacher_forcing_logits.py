import sys
import torch
import numpy as np
import scipy as sp
from ._model import Model
from ..utils import safe_isinstance
from ._text_generation import TextGeneration
from ..utils import get_tokenizer_prefix_suffix

class TeacherForcingLogits(Model):
    def __init__(self, model, tokenizer=None, generation_function_for_target_sentence_ids=None, text_similarity_model=None, text_similarity_tokenizer=None, device=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device 
        self.model = self.to_device(model, device=self.device)
        self.tokenizer = tokenizer
        # assign text generation function
        if safe_isinstance(model,"transformers.PreTrainedModel"):
            if generation_function_for_target_sentence_ids is None:
                self.generation_function_for_target_sentence_ids = TextGeneration(self.model, tokenizer=self.tokenizer, device=self.device)
            else:
                self.generation_function_for_target_sentence_ids = generation_function_for_target_sentence_ids
            self.model_agnostic = False
            self.text_similarity_model = model
            self.text_similarity_tokenizer = tokenizer
            #self.keep_prefix, self.keep_suffix = get_tokenizer_prefix_suffix(self.tokenizer)
        else:
            if generation_function_for_target_sentence_ids is None:
                self.generation_function_for_target_sentence_ids = TextGeneration(self.model, text_similarity_tokenizer=text_similarity_tokenizer, device=self.device)
            else:
                self.generation_function_for_target_sentence_ids = generation_function_for_target_sentence_ids
            self.model_agnostic = True
        # initializing X which is the original input for every new row of explanation
        self.X = None
        self.target_sentence_ids = None

    def __call__(self, masked_X, X):
        output_batch=[]
        for masked_x, x in zip(masked_X, X):
            # update target sentence ids and original input for a new explanation row
            self.update_cache_X(x)
            # pass the masked input from which to generate source sentence ids
            source_sentence_ids = self.get_source_sentence_ids(masked_x)
            logits = self.get_teacher_forced_logits(source_sentence_ids, self.target_sentence_ids)
            logodds = self.get_logodds(logits)
            output_batch.append(logodds)
        return output_batch
