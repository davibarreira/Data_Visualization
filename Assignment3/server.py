import os
import flask
import numpy as np
import time
from PIL import Image
import pandas as pd
import json
import itertools

from tqdm import tqdm

import torch

from sklearn.cluster import KMeans

from flask import Flask
from flask_cors import CORS
import math

from matplotlib import pyplot as plt

import umap

# create Flask app
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
CORS(app)

# TODO load all of the data generated from preprocessing

qt = torch.load('./static/qt.pt')
tensor2_3 = torch.load('./static/tensor2_3.pt')
tensor2_3 = torch.nan_to_num(tensor2_3,nan=0.0) # Ajustando casos nan

tensor3_4 = torch.load('./static/tensor3_4.pt')
tensor3_4 = torch.nan_to_num(tensor3_4,nan=0.0) #  Ajustando casos nan

s23 = tensor2_3.sum(dim=0)
s34 = tensor3_4.sum(dim=0)

images = []
for i in range(0,20):
    images.append(Image.open('./static/image'+str(i)+'.jpeg'))

# number of clusters - feel free to adjust
n_clusters = 9

# these variables will contain the clustering of channels for the different layers
a2_clustering,a3_clustering,a4_clustering = None,None,None

'''
Do not cache images on browser, see: https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
'''
@app.after_request
def add_header(r):
    r.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    r.headers['Pragma'] = 'no-cache'
    r.headers['Expires'] = '0'
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
#

'''
TODO

Implement spectral clustering, given an affinity matrix. You are required to implement this using standard matrix computation libraries, e.g. numpy, for computing a spectral embedding.
You may use k-means once you've obtained the spectral embedding.

NOTE: the affinity matrix should _not_ be symmetric! Nevertheless, eigenvectors will be real, up to numerical precision - so you should cast to real numbers (e.g. np.real).
'''
def spectral_clustering(affinity_mat, n_clusters):
    # code based on https://towardsdatascience.com/spectral-clustering-aba2640c0d5b
    # diagonal matrix
    D = torch.diag(torch.sum(affinity_mat,dim=1))

    # graph laplacian
    L = torch.pow(torch.inverse(D),1/2) @ affinity_mat @ torch.pow(D,1/2)

    # eigenvalues and eigenvectors
    vals, vecs = torch.eig(L,eigenvectors=True)
    vals = vals[:,0]
    # sort these based on the eigenvalues
    vals = vals[torch.argsort(vals,descending=True)]
    vecs = vecs[:,torch.argsort(vals,descending=True)]
    row_norms = torch.norm(vecs[:,0:2],dim=1).unsqueeze(1).repeat(1,2)

    Y = vecs[:,0:2]/row_norms
    
    # kmeans on first three vectors with nonzero eigenvalues
    kmeans = KMeans(n_clusters=n_clusters,random_state=7)
    kmeans.fit(Y[:,1:n_clusters])
    labels = kmeans.labels_
    return labels

'''
TODO

Cluster the channels within each layer.
This should take, as arguments, the two similarity matrices derived from the IoU scores.
Specifically, the first argument is the similarity matrix between channels at layer 2 and channels at layer 3.
The second argument is the similarity matrix between channels at layer 3 and channels at layer 4.

A generalization of spectral biclustering should be performed. More details given in the assignment notebook.
'''
def multiway_spectral_clustering(s23, s34, n_clusters):

    c23 = torch.diag(s23.sum(dim=0))
    r23 = torch.diag(s23.sum(dim=1))

    c34 = torch.diag(s34.sum(dim=0))
    r34 = torch.diag(s34.sum(dim=1))

    s2  = torch.inverse(r23) @ s23 @ torch.transpose(s23 @ torch.inverse(c23),0,1)
    s4  = torch.transpose(s34 @ torch.inverse(c34),0,1) @ torch.inverse(r34) @ s34 
    s3  = torch.transpose(s23 @ torch.inverse(c23),0,1) @ torch.inverse(r23) @ s23 \
            + torch.inverse(r34) @ s34 @ torch.transpose(s34 @ torch.inverse(c34),0,1)
    
    c2 = spectral_clustering(s2,n_clusters)
    c3 = spectral_clustering(s3,n_clusters)
    c4 = spectral_clustering(s4,n_clusters)
    
    return c2, c3, c4


'''
TODO

Given a link selected from the visualization, namely the layer and
clusters at the individual layers, this route should compute the mean
correlation from all channels in the source layer and all channels in the target layer, for each sample.
'''
@app.route('/link_score', methods=['GET','POST'])
def link_score():
    pass
#

'''
TODO

Given a layer (of your choosing), perform max-pooling over space,
giving a vector of activations over channels for each sample.
Perform UMAP to compute a 2D projection.
'''
@app.route('/channel_dr', methods=['GET','POST'])
def channel_dr():
    pass
#

'''
TODO

Compute correlation strength over selected instances, those brushed by the user.
'''
@app.route('/selected_correlation', methods=['GET','POST'])
def selected_correlation():
    pass
#

'''
TODO

Compute correlation strength over all instances.
'''
def correlations_activation(clustering1,clustering2, tensor,n_clusters):
    corr = []
    n      = 0
    for h in list(itertools.combinations(range(n_clusters),2)):
        c1_index = np.where(clustering1==h[0])[0]
        c2_index = np.where(clustering2==h[1])[0]
        corr.append(0)
        Z = tensor.shape[0] * sum(clustering1==h[0]) * sum(clustering2==h[1])
        for j in c1_index:
            for k in c2_index:
                corr[n] += tensor[:,j,k].sum()
        corr[n] = corr[n]/Z
        n+=1
    return corr

@app.route('/activation_correlation_clustering', methods=['GET'])
def activation_correlation_clustering():
    # Calculate the correlations
    corr23 = correlations_activation(a2_clustering,a3_clustering,tensor2_3,n_clusters)
    corr34 = correlations_activation(a3_clustering,a4_clustering,tensor3_4,n_clusters)

    # Turn tensor into float
    corr23 = [i.item() for i in corr23]
    corr34 = [i.item() for i in corr34]

    # Create dataset with the correlation and links
    cline23 = []
    cline34 = []
    for h in list(itertools.combinations(range(n_clusters),2)):
        cline23.append((h[0],h[1]))
        cline34.append((h[0],h[1]))
    links23 = pd.DataFrame(cline23,columns=["cluster_input","cluster_output"])
    links23["corr"] = corr23
    links34 = pd.DataFrame(cline34,columns=["cluster_input","cluster_output"])
    links34["corr"] = corr34

    return json.dumps({
        "link23":links23.to_dict(orient='records'),
        "link34":links34.to_dict(orient='records'),
        })


'''
TODO

In the main, before running the server, run clustering,
store results in variables a2_clustering, a3_clustering, a4_clustering
'''
if __name__=='__main__':
    a2_clustering,a3_clustering,a4_clustering = multiway_spectral_clustering(s23,s34,n_clusters)

    app.run()
#
