from transformers import TrainerCallback

class SaveTokenizerCallback(TrainerCallback):
    def __init__(self, tokenizer, save_directory):
        self.tokenizer = tokenizer
        self.save_directory = save_directory

    def on_epoch_end(self, args, state, control, **kwargs):
        self.tokenizer.save_pretrained(self.save_directory)
