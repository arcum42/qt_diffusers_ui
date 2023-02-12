# This Python file uses the following encoding: utf-8
import sys
import os
import torch
import json
import random

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QFile, Slot
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QImage, QPixmap
from pathlib import Path

from diffusers import StableDiffusionPipeline
from diffusers import LMSDiscreteScheduler
from diffusers import EulerAncestralDiscreteScheduler
from diffusers import EulerDiscreteScheduler
from diffusers import DDPMScheduler
from diffusers import HeunDiscreteScheduler
from diffusers import DDIMScheduler
from diffusers import PNDMScheduler
from diffusers import DPMSolverSinglestepScheduler
from diffusers import DPMSolverMultistepScheduler

global config
global pipe

scheduler_list = {
    "euler a", "euler", "LMS", "DDPM", "heun", "DDIM", "PNDM", "DPM Solver single", "DPM Solver multi", "DPM Solver++ single", "DPM Solver++ multi"
}

def setJSONToDefaults():
    global config
    config = {
    'modelPath' : 'model',
    'imagePath' : 'images',
    'modelName' : 'waifu-diffusion',
    'imageName' : 'img.png',
    'width' :  512,
    'height' : 512,
    'steps' : 20,
    'cfg' : 7,
    'local' : True,
    'seed' : -1,
    'safety' : True,
    'remote-model' : 'hakurei/waifu-diffusion',
    'scheduler' : 'euler a'
    }

def loadJSON():
    global config

    setJSONToDefaults()
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print("Config file not found.")

def saveJSON():
    with open('config.json', 'w') as f:
        json.dump(config, f, indent = 4, sort_keys=True)
    print ("Saved configuration.")

#https://huggingface.co/docs/diffusers/v0.12.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline

def saveLocalModel():
    global pipe
    modelName = config['remote-model'].split('/')[1]
    newModelPath = Path(config['modelPath'])/modelName
    if not newModelPath.exists():
        pipe.save_pretrained(save_directory=newModelPath, safe_serialization=True)

def addSchedulers():
    for x in scheduler_list:
        window.schedulerBox.addItem(x)

def setScheduler():
    global pipe
    match config['scheduler']:
        case "euler a":
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        case "euler":
            pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

        case "LMS":
            pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

        case "DDPM":
            pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        case "heun":
            pipe.scheduler = HeunDiscreteScheduler.from_config(pipe.scheduler.config)

        case "DDIM":
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

        case "PNDM":
            pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

        case "DPM Solver single": #1S, maybe?
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver", solver_order=2)

        case "DPM Solver multi": #2M?
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver", solver_order=2)

        case "DPM Solver++ single": #1S, maybe?
            pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2)

        case "DPM Solver++ multi": #2M?
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", solver_order=2)


def initModel():
    global pipe

    modelFilename = Path(Path(config['modelPath'])/config['modelName']).absolute()
    if config['local'] == True:
        ourModel = modelFilename
    else:
        ourModel = config['remote-model']
    safety = config['safety']
    print(f"Loading model {ourModel}. Safety is {safety}.")
    if safety == True:
        pipe = StableDiffusionPipeline.from_pretrained(ourModel, resume_download=True)
    else:
        pipe = StableDiffusionPipeline.from_pretrained(ourModel, safety_checker=None, resume_download=True)

    if config['local'] == False:
        print("Saving remote model locally.")
        saveLocalModel()

    setScheduler()
    pipe.to("cuda")
    pipe.enable_attention_slicing()

def pipeCallback(step: int, timestep: int, latents: torch.FloatTensor):
    window.generationProgress.setMinimum(1)
    window.generationProgress.setMaximum(config["steps"])
    window.generationProgress.setValue(step)

@Slot()
def generateArt(self):
    global pipe

    prompt = window.promptText.toPlainText()
    negPrompt = window.negPromptText.toPlainText()
    modelFilename = Path(Path(config['modelPath'])/config['modelName']).absolute()
    imageFilename = Path(Path(config['imagePath'])/config['imageName']).absolute()
    if config['local'] == True:
        ourModel = modelFilename
    else:
        ourModel = config['remote-model']

    current_seed = config['seed']
    if current_seed == -1:
        current_seed = random.randrange(2147483647)

    generator = torch.Generator(device="cuda").manual_seed(current_seed)
    #generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(4)]
    print(f"Prompt: '{prompt}', Negative prompt: '{negPrompt}', Steps: {config['steps']}, CFG: {config['cfg']}, width: {config['width']}, height: {config['height']}, seed: {current_seed}. Model is '{ourModel}'. Saving to '{imageFilename}'.")

    if not negPrompt:
        image = pipe(
        prompt=prompt,
        generator=generator,
        guidance_scale=config['cfg'],
        num_inference_steps=config['steps'],
        height=config['height'],
        width=config['width'],
        callback=pipeCallback
        ).images[0]
    else:
        #seed, batch_size
        image = pipe(
        prompt=prompt,
        generator=generator,
        negative_prompt=negPrompt,
        guidance_scale=config['cfg'],
        num_inference_steps=config['steps'],
        height=config['height'],
        width=config['width'],
        callback=pipeCallback
        ).images[0]
    window.generationProgress.setValue(config["steps"])
    image.save(imageFilename)

    localImage = QImage(imageFilename)
    localPixmap = QPixmap(localImage)
    window.aiArt.setPixmap(localPixmap)

@Slot()
def refreshModelList(self):
    global config

    window.modelsComboBox.clear()
    modelDirs = next(os.walk(Path(config['modelPath'])))[1]
    print(modelDirs)
    for x in modelDirs:
        print("Found {}.".format(x))
        window.modelsComboBox.addItem(x)
    window.modelsComboBox.setCurrentText(config["modelName"])
    config["modelName"] = window.modelsComboBox.currentText()

@Slot()
def modelChanged(self):
    config["modelName"] = window.modelsComboBox.currentText()
    initModel()
    print("Model set to {}.".format(config["modelName"]))

@Slot()
def schedulerChanged(self):
    config["scheduler"] = window.schedulerBox.currentText()
    setScheduler()
    print("Scheduler set to {}.".format(config["scheduler"]))

@Slot()
def changeImageName():
    config["imageName"] = window.imageFilenameText.text()
    print("Image filename will be {}.".format(config["imageName"]))

@Slot()
def changeModelPath():
    config["modelPath"] = Path(window.modelPathText.text()).absolute()
    print("Model path is now {}.".format(config["modelPath"]))
    refreshModelList(0)

@Slot()
def changeImagePath():
    config["imagePath"] = Path(window.imagePathText.text()).absolute()
    print("Image path is now {}.".format(config["imagePath"]))

@Slot()
def updateCFG():
    config['cfg'] = window.cfgSpin.value()

@Slot()
def updateSteps():
    config['steps'] = window.stepsSpin.value()

@Slot()
def updateWidth():
    config['width'] = window.widthSpin.value()

@Slot()
def updateHeight():
    config['height'] = window.heightSpin.value()

@Slot()
def updateSeed():
    config['seed'] = window.seedSpin.value()

@Slot()
def safety_dance():
    config['safety'] = window.safetyCheck.isChecked()
    print("Reloading model...")
    initModel()

@Slot()
def close_down():
    saveJSON()

@Slot()
def checkLocal():
    config['local'] = window.localRadio.isChecked()
    initModel()

@Slot()
def changeRemoteModel():
    config['remote-model'] = window.remoteUrlText.text()

def set_ui_from_config():
    # Load configs into the ui.
    window.imageFilenameText.setText(config['imageName'])
    window.modelPathText.setText(config['modelPath'])
    window.imagePathText.setText(config['imagePath'])
    window.widthSpin.setValue(config['width'])
    window.heightSpin.setValue(config['height'])
    window.stepsSpin.setValue(config['steps'])
    window.cfgSpin.setValue(config['cfg'])
    window.seedSpin.setValue(config['seed'])
    window.safetyCheck.setChecked(config['safety'])
    window.remoteUrlText.setText(config['remote-model'])
    if config['local'] == True:
        window.localRadio.setChecked(True)
    else:
        window.remoteRadio.setChecked(True)

    # Iterate through and list all the models in the model directory.
    refreshModelList(0)
    addSchedulers()
    window.schedulerBox.setCurrentText(config["scheduler"])

def connect_ui():
    # Connect all the widgets
    window.generateButton.clicked.connect(generateArt)
    window.modelsRefreshButton.clicked.connect(refreshModelList)

    window.modelsComboBox.currentTextChanged.connect(modelChanged)
    window.imageFilenameText.editingFinished.connect(changeImageName)
    window.modelPathText.editingFinished.connect(changeModelPath)
    window.imagePathText.editingFinished.connect(changeImagePath)
    window.widthSpin.valueChanged.connect(updateWidth)
    window.heightSpin.valueChanged.connect(updateHeight)
    window.stepsSpin.valueChanged.connect(updateSteps)
    window.cfgSpin.valueChanged.connect(updateCFG)
    window.seedSpin.valueChanged.connect(updateSeed)
    window.safetyCheck.stateChanged.connect(safety_dance)
    window.localRadio.toggled.connect(checkLocal)
    window.remoteUrlText.editingFinished.connect(changeRemoteModel)
    window.schedulerBox.currentTextChanged.connect(schedulerChanged)
    app.aboutToQuit.connect(close_down)

# Start of main function
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Load our config file.
    random.seed()
    loadJSON()

    # Load the ui file.
    loader = QUiLoader()
    ui_file = QFile("mainwindow.ui")
    ui_file.open(QFile.ReadOnly)
    window = loader.load(ui_file)
    ui_file.close()

    window.setWindowTitle("QT Diffusers UI")
    window.theTabs.setTabEnabled(1, False)
    window.theTabs.setTabEnabled(2, False)

    set_ui_from_config()
    connect_ui()

    initModel()
    window.show()

    #Save configuration.
    saveJSON()

    sys.exit(app.exec())
