import copy
import os
import gc
import re
import json
import queue

# from tqdm.contrib.telegram import tqdm, trange
from tqdm import tqdm, trange
import logging
import time
import torch
from transformers import TextDataset, DataCollatorForLanguageModeling, GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import DataLoader

from accelerate import Accelerator
from transformers import AdamW, AutoModelForSequenceClassification, get_scheduler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer, TrainingArguments

all_groups = set()
order_queue = queue.PriorityQueue()
NN = None
epochs_list = [0, 1, 10, 100]


logging.basicConfig(format="%(asctime)s %(message)s", handlers=[logging.FileHandler(
    f"/home/logs/text_gen_log_model.txt", mode="a")], datefmt="%I:%M:%S %p", level=logging.INFO)


class NeuralNetwork:
    def __init__(self):
        self.DEVICE = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"
        self.tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        self.model = GPT2LMHeadModel.from_pretrained(checkpoint)
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

    def tune(self, group_id, epochs, checkpoint_path="weights/", train_dataset_path="train_test_datasets/train", test_dataset_path="train_test_datasets/test"):
        logging.info("Tuning")
        logging.info(f"Torch Cuda is available: {torch.cuda.is_available()}")
        train_dataset_path = train_dataset_path + str(group_id)
        test_dataset_path = test_dataset_path + str(group_id)
        self.load_dataset(train_dataset_path, test_dataset_path)

        train_loader = DataLoader(
            self.train_dataset, shuffle=True, batch_size=1, collate_fn=self.data_collator)
        test_loader = DataLoader(
            self.test_dataset, batch_size=1, collate_fn=self.data_collator)

        optimizer = AdamW(self.model.parameters(), lr=3e-5)
        save_checkpoint_path = checkpoint_path + str(group_id) + ".pt"

        accelerator = Accelerator()
        train_dl, test_dl, self.model, optimizer = accelerator.prepare(
            train_loader, test_loader, self.model, optimizer
        )
        # with open("/home/config.json", "r", encoding="UTF-8") as f:
        #     data = json.load(f)
        #     token = data["tg_token"]
        #     chat_id = data["tg_chat_id"]
        training_args = TrainingArguments(logging_dir="logs/",
                                          output_dir="weights/",
                                          logging_first_step=True,
                                          logging_steps=1,
                                          num_train_epochs=epochs,
                                          per_device_train_batch_size=32,
                                          per_device_eval_batch_size=32,
                                          warmup_steps=10,
                                          gradient_accumulation_steps=16,
                                          )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
        )
        logging.info(
            f"Start tuning from train dataset: {train_dataset_path}")
        try:
            if epochs == 0:
                logging.info(f"trainer_trained {trainer.args.num_train_epochs} epochs")
                if os.path.isfile(save_checkpoint_path):
                    os.remove(save_checkpoint_path)
                trainer.save_model(output_dir=checkpoint_path + str(group_id))
                torch.save(
                    {'model_state_dict': self.model.state_dict(), },
                    checkpoint_path + str(group_id) + "-trained.pt"
                )
                logging.info(f"Weights saved: {checkpoint_path + str(group_id)}-trained.pt")
                return
            trainer.train()
            logging.info(f"trainer_trained {trainer.args.num_train_epochs} epochs")
            if os.path.isfile(save_checkpoint_path):
                os.remove(save_checkpoint_path)
            trainer.save_model(output_dir=checkpoint_path + str(group_id))
            torch.save(
                {'model_state_dict': self.model.state_dict(), },
                checkpoint_path + str(group_id) + "-trained.pt"
            )
            logging.info(f"Weights saved: {checkpoint_path + str(group_id)}-trained.pt")
        except Exception as e:
            logging.error(
                f"An error occured: {e}, dataset: {train_dataset_path}")
        # os.rename(save_checkpoint_path, checkpoint_path +
        #           str(self.group_id) + "-trained.pt")

    def load_weights(self, group_id, checkpoint_path="weights/"):
        logging.info(f"Loading weights: {checkpoint_path}")
        checkpoint_path = checkpoint_path + str(group_id) + "-trained.pt"
        checkpoint = torch.load(checkpoint_path, map_location=self.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("weights loaded")


def main():
    global NN
    logging.info("Creating primary NeuralNetwork")
    NN = NeuralNetwork()
    logging.info("Primary NeuralNetwork is ready")
    global all_groups
    global order_queue
    while True:
        time.sleep(5)
        logging.info("Scanning for new datasets")
        directory = 'train_test_datasets'
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            if os.path.isfile(f) and filename.startswith("train"):
                group_id = int(filename[5:])
                if group_id not in all_groups:
                    order_queue.put((0, group_id))
                    logging.info(f"Found new group: {group_id}")
        if order_queue.qsize() == 0:
            continue

        epochs, group_id = order_queue.get()
        logging.info(f"Start training for group: {group_id}; epochs: {epochs}")
        all_groups.add(group_id)
        tmp_nn = copy.deepcopy(NN)
        tmp_nn.tune(group_id, epochs)
        del tmp_nn
        gc.collect()
        if epochs != epochs_list[-1]:
            next_epochs = epochs_list[epochs_list.index(epochs) + 1]
            logging.info(f"Next epochs: {next_epochs}")
            order_queue.put((next_epochs, group_id))


if __name__ == "__main__":
    main()
