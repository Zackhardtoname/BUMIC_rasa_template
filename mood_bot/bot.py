# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import warnings

from rasa_core.actions import Action
from rasa_core.agent import Agent
from rasa_core.channels.console import ConsoleInputChannel
from rasa_core.events import SlotSet
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
import policy

logger = logging.getLogger(__name__)

class RestaurantAPI(object):
    def search(self, info):
        return "papi's pizza place"

def train_nlu(nlu_data_dir, config_dir, nlu_model_dir):
    from rasa_nlu.training_data import load_data
    from rasa_nlu.model import Trainer
    from rasa_nlu import config

    training_data = load_data(nlu_data_dir)
    trainer = Trainer(config.load(config_dir))
    trainer.train(training_data)
    model_directory = trainer.persist(nlu_model_dir, project_name="nlu", fixed_model_name="nlu")
    return model_directory

def train_dialogue(domain_file, dia_data_file, nlu_model_dir, dia_model_dir):
    from rasa_core.featurizers import (MaxHistoryTrackerFeaturizer,BinarySingleStateFeaturizer)
    featurizer = MaxHistoryTrackerFeaturizer(BinarySingleStateFeaturizer(), max_history=5)
    agent = Agent(domain_file, policies=[MemoizationPolicy(max_history=5), KerasPolicy(featurizer), policy.fallback], \
                  interpreter=RasaNLUInterpreter(nlu_model_dir))
    training_data = agent.load_data(dia_data_file)
    agent.train(training_data,
                epochs = 400,
                batch_size = 100,
                # max_history=5,
                validation_split = 0.2)
    #         augmentation_factor=50,
    agent.persist(dia_model_dir)
    return agent


def learn_interactive(nlu_model_dir, input_channel=ConsoleInputChannel(),
                      domain_file="domain.yml",
                      training_data_file="dia_data/stories.md"):
    interpreter = RasaNLUInterpreter(nlu_model_dir)
    agent = Agent(domain_file,
                  policies=[MemoizationPolicy(), KerasPolicy(), policy.fallback],
                  interpreter=interpreter)

    training_data = agent.load_data(training_data_file)
    agent.train_online(training_data,
                       input_channel=input_channel,
                       # max_history=2,
                       batch_size=50,
                       epochs=200,
                       max_training_samples=300)
    return agent


def classify_intent(nlu_model_dir):
    from rasa_nlu.model import Interpreter
    import json
    # where model_directory points to the model folder
    interpreter = Interpreter.load(nlu_model_dir)
    while True:
        res = interpreter.parse(input('Me: '))
        print(json.dumps(res, ensure_ascii = False, indent=2))

def run(nlu_model_dir="models/current/nlu", dia_model_dir='models/dialogue', serve_forever=True):
    agent = Agent.load(dia_model_dir,
                       interpreter=RasaNLUInterpreter(nlu_model_dir))
    if serve_forever:
        agent.handle_channel(ConsoleInputChannel())
    return agent

if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)

    choices = ["train_nlu", "test_train_nlu", "train_dialogue", "run", "classify_intent", "learn_interactive", "test_train_nlu, train_dialogue, run"]
    #file paths
    config_dir = "config.yml" #configs of pipelines
    nlu_data_dir = "nlu_data/formal" #training data for nlu
    dia_data_file = "dia_data/stories.md" #scenarios
    nlu_model_dir = "models/nlu/nlu" #the path of the trained nlu model
    dia_model_dir = 'models/dialogue' #the path of the trained dialogue model
    domain_file = "domain.yml"      #domain where intents, entities, and slots are defined

    test_nlu_file = r"./nlu_data/test"  #test training data for nlu

    while True:
        print('Options:', choices)
        task = input('me > ')
        # exit
        if task == 'quit':
            break
        # decide what to do based on first parameter of the script
        if task == "train_nlu":
            train_nlu(nlu_data_dir=nlu_data_dir, \
                      config_dir=config_dir, nlu_model_dir='models')
        elif task == "test_train_nlu": #Note this one overwrites the old nlu model
            train_nlu(nlu_data_dir=test_nlu_file, \
                      config_dir=config_dir, nlu_model_dir='models')
        elif task == "train_dialogue":
            train_dialogue(domain_file=domain_file, \
                           dia_data_file=dia_data_file, \
                           nlu_model_dir=nlu_model_dir, dia_model_dir=dia_model_dir)
        elif task == "classify_intent":
            classify_intent(nlu_model_dir=nlu_model_dir)
        elif task == "run":
            run(nlu_model_dir= nlu_model_dir, dia_model_dir=dia_model_dir)
        elif task == "learn_interactive":
            learn_interactive(nlu_model_dir=nlu_model_dir)
        elif task == 'test_train_nlu, train_dialogue, run':
            train_nlu(nlu_data_dir=test_nlu_file, \
                      config_dir=config_dir, nlu_model_dir='models')
            train_dialogue(domain_file=domain_file, \
                           dia_data_file=dia_data_file, \
                           nlu_model_dir=nlu_model_dir, dia_model_dir=dia_model_dir)
            run(nlu_model_dir= nlu_model_dir, dia_model_dir=dia_model_dir)
        else:
            warnings.warn("Need to pass either 'train-nlu', 'train-dialogue' or "
                          "'run' to use the script.")