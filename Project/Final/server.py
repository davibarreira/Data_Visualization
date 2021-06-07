import os
import flask
from flask import request
import numpy as np
import argparse
import json
import csv

from flask import Flask
from flask_cors import CORS
import math

from sklearn import cluster
import pandas as pd

# create Flask app
app = Flask(__name__)
CORS(app)

# --- these will be populated in the main --- #
@app.route("/")
def hello():
    return "Hello World!"


if __name__=='__main__':

    app.run(debug=True)
#

