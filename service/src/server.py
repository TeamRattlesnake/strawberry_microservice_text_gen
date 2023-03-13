import os
import copy
import logging
from multiprocessing import Process
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.utils import get_openapi
from models import ResponseModel, AddGroupModel, GenerateModel
from logic import NeuralNetwork


logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%I:%M:%S %p", level=logging.INFO)

WEIGHTS_DIR = "weights"
TRAIN_TEST_DATASETS_DIR = "train_test_datasets"
CONTENT_DIR = "content"

app = FastAPI()

app.mount("/content", StaticFiles(directory="content"), name="content")

NN = None
process_pool = {}

DESCRIPTION = """
Микросервис для Strawberry

"""


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Strawberry Microservice",
        version="0.0.5",
        description=DESCRIPTION,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.on_event("startup")
def startup():
    '''Функция, запускающаяся при старте сервера'''
    logging.info("Server started")
    if not os.path.exists("weights"):
        os.makedirs(WEIGHTS_DIR)
        logging.info(f"Created {WEIGHTS_DIR} directory")
    if not os.path.exists("train_test_datasets"):
        os.makedirs(TRAIN_TEST_DATASETS_DIR)
        logging.info(f"Created {TRAIN_TEST_DATASETS_DIR} directory")
    global NN
    logging.info("Creating primary NeuralNetwork")
    NN = NeuralNetwork()
    logging.info("Primary NeuralNetwork is ready")


@app.post("/add_group", response_model=ResponseModel)
async def add_group(data: AddGroupModel):
    group_id = data.group_id
    texts = data.texts
    logging.info(f"Adding group {group_id}")
    try:
        if len(texts) == 0:
            raise ValueError("Empty texts")
        if (not os.path.exists(f"{WEIGHTS_DIR}/{group_id}-trained.pt")) and (not os.path.exists(f"{WEIGHTS_DIR}/{group_id}.pt")):
            f = open(f"{WEIGHTS_DIR}/{group_id}.pt", 'x')
            f.close()
            tmp_nn = copy.deepcopy(NN)
            tmp_nn.group_id = group_id
            global process_pool
            p = Process(target=tmp_nn.tune, args=(texts,))
            p.start()
            process_pool[group_id] = p
            # tmp_nn.tune(texts)
            return ResponseModel(result="OK")
        return ResponseModel(result="NO")
    except Exception as e:
        logging.error(e)
        return ResponseModel(result="ERROR")


@app.post("/generate", response_model=ResponseModel)
async def generate(data: GenerateModel):
    group_id = data.group_id
    hint = data.hint
    logging.info(f"Generating content for group {group_id}")
    try:
        if os.path.exists(f"{WEIGHTS_DIR}/{group_id}-trained.pt"):
            tmp_nn = copy.deepcopy(NN)
            tmp_nn.load_weights(group_id)
            result = tmp_nn.generate(hint)
            if process_pool.get(group_id):
                (process_pool[group_id]).join()
                del process_pool[group_id]
            return ResponseModel(result=result)
        return ResponseModel(result="NO")
    except Exception as e:
        logging.error(e)
        return ResponseModel(result="ERROR")


@app.get("/check_status", response_model=ResponseModel)
async def check_status(group_id: int):
    logging.info(f"Cheking status for group {group_id}")
    try:
        if os.path.exists(f"{WEIGHTS_DIR}/{group_id}-trained.pt"):
            return ResponseModel(result="OK")
        return ResponseModel(result="NO")
    except Exception as e:
        logging.error(e)
        return ResponseModel(result="ERROR")
