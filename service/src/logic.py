import os
import time
import re
import json

# from tqdm.contrib.telegram import tqdm, trange
from tqdm import tqdm, trange
import logging
import time
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments


logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/text_gen_server_log_{time.ctime()}.txt", mode="a", encoding="UTF-8")], datefmt="%I:%M:%S %p", level=logging.INFO)


def clean_string(string):
    permitted_chars = "^0-9A-Za-zА-Яа-яёЁ!,:;.!?/@#()*+-"
    string = re.sub(f"[{permitted_chars}]+", " ", string)
    return string


class NeuralNetwork:
    def __init__(self, group_id=0):
        #device_string = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(
            f"Torch uses: cpu (поставил на время пока не придумаем решение)")
        self.DEVICE = torch.device("cpu")
        checkpoint = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
        self.group_id = group_id
        self.train_dataset = None
        self.test_dataset = None
        self.data_collator = None

    def get_length_param(self, text: str) -> str:
        tokens_count = len(self.tokenizer.encode(text))
        if tokens_count <= 15:
            len_param = '1'
        elif tokens_count <= 50:
            len_param = '2'
        elif tokens_count <= 256:
            len_param = '3'
        else:
            len_param = '-'
        return len_param

    def build_text_file(self, texts: list[str], dest_path: str = "train_test_datasest/"):
        logging.info("Building text file")
        with open(dest_path, 'w') as f:
            for text in texts:
                post_text = re.sub(r"\n", ". ", text)
                if len(post_text) == 0 or type(post_text) != str:
                    logging.info(f"Empty text or not text at all: {post_text}")
                    continue
                length = self.get_length_param(post_text)
                f.write(f"|{length}|{post_text}{self.tokenizer.eos_token}\n")

    def load_dataset(self, train_path, test_path):
        logging.info(f"Loading datasets: {train_path}, {test_path}")
        self.train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=train_path,
            block_size=64
        )

        self.test_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=test_path,
            block_size=64
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )

    def tune(self, texts, checkpoint_path="weights/", train_dataset_path="train_test_datasets/train", test_dataset_path="train_test_datasets/test"):
        texts = [clean_string(string) for string in texts]
        logging.info("Tuning")
        logging.info(f"Torch Cuda is available: {torch.cuda.is_available()}")
        train_dataset_path = train_dataset_path + str(self.group_id)
        test_dataset_path = test_dataset_path + str(self.group_id)
        self.build_text_file(
            texts[int(len(texts)*0.1):], dest_path=train_dataset_path)
        self.build_text_file(
            texts[:int(len(texts)*0.1)], dest_path=test_dataset_path)
        self.load_dataset(train_dataset_path, test_dataset_path)

    def load_weights(self, group_id, checkpoint_path="weights/"):
        logging.info(f"Loading weights: {checkpoint_path}")
        checkpoint_path = checkpoint_path + str(group_id) + "-trained.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("weights loaded")

    def generate(self, hint):
        logging.info(f"Generating with hint: {hint}")
        hint = clean_string(hint)
        text = "<|startoftext|>" + hint
        input_ids = self.tokenizer.encode(
            text, return_tensors="pt").to(self.DEVICE)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(input_ids,
                                      do_sample=True,
                                      temperature=1.9,
                                      top_k=50,
                                      top_p=0.9,
                                      max_length=150,
                                      num_return_sequences=1,
                                      eos_token_id=self.tokenizer.eos_token_id,
                                      pad_token_id=self.tokenizer.pad_token_id,
                                      )
        generated_text = list(map(self.tokenizer.decode, out))[0]
        generated_text = generated_text.replace("<|startoftext|>", "")
        generated_text = generated_text.split("</s>")[0].strip()
        logging.info(f"Generation for hint is over: {generated_text}")
        return generated_text
