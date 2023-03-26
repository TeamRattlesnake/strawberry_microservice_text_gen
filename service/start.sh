#!/bin/bash

uvicorn --app-dir src --host 0.0.0.0 --port 20001 server:app &
python3 training_model/model.py
