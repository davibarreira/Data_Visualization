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

# list of attribute names of size m
attribute_names=None

# a 2D numpy array containing binary attributes - it is of size n x m, for n paintings and m attributes
painting_attributes=None

# a list of epsiode names of size n
episode_names=None

# a list of painting image URLs of size n
painting_image_urls=None

projection = np.array([0,0])
x_loadings = []
y_loadings = []

'''
This will return an array of strings containing the episode names -> these should be displayed upon hovering over circles.
'''
@app.route('/get_episode_names', methods=['GET'])
def get_episode_names():
    return flask.jsonify(episode_names)
#

'''
This will return an array of URLs containing the paths of images for the paintings
'''
@app.route('/get_painting_urls', methods=['GET'])
def get_painting_urls():
    return flask.jsonify(painting_image_urls)
#

'''
TODO: implement PCA, this should return data in the same format as you saw in the first part of the assignment:
    * the 2D projection
    * x loadings, consisting of pairs of attribute name and value
    * y loadings, consisting of pairs of attribute name and value
'''
@app.route('/initial_pca', methods=['GET'])
def initial_pca():
    data_centered = painting_attributes.copy() 
    data_centered -= np.mean(data_centered, axis=0) 
    C = np.cov(data_centered.T)
    w,v = np.linalg.eig(C)
    eig_idx = np.argpartition(w, -2)[-2:]
    eig_idx = eig_idx[np.argsort(-w[eig_idx])]
    pca_components = -v[:,eig_idx]
    x_loadings = pd.DataFrame(zip(attribute_names,pca_components[:,0]),
                                    columns=['attribute','loading']).to_dict(orient='records')
    y_loadings = pd.DataFrame(zip(attribute_names,pca_components[:,1]),
                                    columns=['attribute','loading']).to_dict(orient='records')

    print(x_loadings)
    proj_data = []
    for row in data_centered:
        proj_data.append(pca_components.T @ row)
    projection = np.array(proj_data).copy()

    return json.dumps({"projection": projection.tolist(),
                            "loading_x": x_loadings,
                            "loading_y": y_loadings})
#

'''
TODO: implement ccPCA here. This should return data in the same format as initial_pca above.
It will take in a list of data items, corresponding to the set of items selected in the visualization.
This can be acquired from `flask.request.json`. This should be a list of data item indices - the **target set**.
The alpha value, from the paper, should be set to 1.1 to start, though you are free to adjust this parameter.
'''
def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


@app.route('/ccpca', methods=['GET','POST'])
def ccpca():
    global projection
    global x_loadings
    global y_loadings

    if request.method == 'POST':
        content = request.get_json()
        print(type(content))
        ids = list(range(len(painting_attributes)))
        marked = [int(i) for i in content['marked_data']]
        unmarked = Diff(ids,marked)

        alpha = 1.1
        data_centered = painting_attributes.copy() 
        data_centered -= np.mean(data_centered, axis=0) 
        C = np.cov(data_centered.T)

        cluster_y = data_centered[unmarked,:]
        cluster_y -= np.mean(cluster_y, axis=0) 

        Cy = np.cov(cluster_y.T)
        Cr = C - alpha*Cy

        w,v = np.linalg.eig(Cr)
        eig_idx = np.argpartition(w, -2)[-2:]
        eig_idx = eig_idx[np.argsort(-w[eig_idx])]
        pca_components = v[:,eig_idx]
        x_loadings = pd.DataFrame(zip(attribute_names,pca_components[:,0]),
                                        columns=['attribute','loading']).to_dict(orient='records')
        y_loadings = pd.DataFrame(zip(attribute_names,pca_components[:,1]),
                                        columns=['attribute','loading']).to_dict(orient='records')

        proj_data = []
        for row in data_centered:
            proj_data.append(pca_components.T @ row)
        projection = np.array(proj_data).copy()

    return json.dumps({"projection": projection.tolist(),
                            "loading_x": x_loadings,
                            "loading_y": y_loadings})
  
#

'''
TODO: run kmeans on painting_attributes, returning data in the same format as in the first part of the assignment.
Namely, an array of objects containing the following properties:
    * label - the cluster label
    * id: the data item's id, simply its index
    * attribute: the attribute name
    * value: the binary attribute's value
'''
@app.route('/kmeans', methods=['GET'])
def kmeans():
    kmeans = cluster.KMeans(6)
    kmeans.fit(painting_attributes)
    df = pd.DataFrame(painting_attributes,columns=attribute_names)
    df["id"] = df.index
    df['label'] = kmeans.labels_
    df = df.melt(id_vars=['id','label']).sort_values(by='id').to_json(orient='records')
    return df
#

if __name__=='__main__':
    painting_image_urls = json.load(open('painting_image_urls.json','r'))
    attribute_names = json.load(open('attribute_names.json','r'))
    episode_names = json.load(open('episode_names.json','r'))
    painting_attributes = np.load('painting_attributes.npy')

    app.run(debug=True)
#
