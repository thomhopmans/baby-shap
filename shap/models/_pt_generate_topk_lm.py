from ._generate_topk_lm import GenerateTopKLM
from ..utils import safe_isinstance, record_import_error
from ..utils.transformers import MODELS_FOR_CAUSAL_LM

try:
    import torch
except ImportError as e:
    record_import_error("torch", "Torch could not be imported!", e)

class PTGenerateTopKLM(GenerateTopKLM):
    def __init__(self, model, tokenizer, k=10, generation_function_for_topk_token_ids=None, device=None):
        """ Generates scores (log odds) for the top-k tokens for Causal/Masked LM for PyTorch models.

        This model inherits from GenerateTopKLM. Check the superclass documentation for the generic methods the library implements for all its model.

        Parameters
        ----------
        model: object or function
            A object of any pretrained transformer model which is to be explained.

        tokenizer: object
            A tokenizer object(PreTrainedTokenizer/PreTrainedTokenizerFast).

        generation_function_for_topk_token_ids: function
            A function which is used to generate top-k token ids. Log odds will be generated for these custom token ids.

        Returns
        -------
        numpy.array
            The scores (log odds) of generating top-k token ids using the model.
        """
        super(PTGenerateTopKLM, self).__init__(model, tokenizer, k, generation_function_for_topk_token_ids, device)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if self.device is None else self.device 
        self.model = model.to(self.device)
        self.generate_topk_token_ids = generation_function_for_topk_token_ids if generation_function_for_topk_token_ids is not None else self.generate_topk_token_ids

    def get_sentence_ids(self, X):
        """ The function tokenizes sentence.

        Parameters
        ----------
        X: string
            X is a sentence.

        Returns
        -------
        torch.Tensor
            Tensor of sentence ids.
        """
        sentence_ids = torch.tensor([self.tokenizer.encode(X)], device=self.device).to(torch.int64)
        return sentence_ids

    def generate_topk_token_ids(self, X):
        """ Generates top-k token ids for Causal/Masked LM.

        Parameters
        ----------
        X: string
            Input(Text) for an explanation row.

        Returns
        -------
        torch.Tensor
            A tensor of top-k token ids.
        """
        
        sentence_ids = self.get_sentence_ids(X)
        logits = self.get_lm_logits(sentence_ids)
        topk_tokens_ids = torch.topk(logits, self.k, dim=1).indices[0]
        return topk_tokens_ids

    def get_lm_logits(self, sentence_ids):
        """ Evaluates a Causal/Masked LM model and returns logits corresponding to next word/masked word.

        Parameters
        ----------
        source_sentence_ids: torch.Tensor of shape (batch size, len of sequence)
            Tokenized ids fed to the model.

        Returns
        -------
        torch.Tensor
            Logits corresponding to next word/masked word.
        """
        # set model to eval mode
        self.model.eval()
        if safe_isinstance(self.model, MODELS_FOR_CAUSAL_LM):
            if sentence_ids.shape[1]==0:
                if hasattr(self.model.config,"bos_token_id") and self.model.config.bos_token_id is not None:
                    sentence_ids = (
                        torch.ones((sentence_ids.shape[0], 1), dtype=sentence_ids.dtype, device=sentence_ids.device)
                        * self.model.config.bos_token_id
                    )
                else:
                    raise ValueError(
                    "Context ids (source sentence ids) are null and no bos token defined in model config"
                )
            # generate outputs and logits
            with torch.no_grad():
                outputs = self.model(sentence_ids, return_dict=True)
            # extract only logits corresponding to target sentence ids
            logits=outputs.logits.detach().cpu()[:,sentence_ids.shape[1]-1,:]
            del outputs    
        return logits