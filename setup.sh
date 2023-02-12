#/bin/sh

python -m venv venv

source venv/bin/activate

pip install pyside6
pip install --upgrade diffusers[torch] transformers accelerate scipy safetensors

