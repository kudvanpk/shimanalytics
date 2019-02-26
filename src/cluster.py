#!/usr/bin/env python

from __future__ import print_function
import time
import signal
import os
import sys
import argparse
import datetime
import requests
import exceptions
import json
import threading
import networkx as nx
import matplotlib
matplotlib.use('Agg')
from hac import GreedyAgglomerativeClusterer
import pandas as pd
import csv
import json
import logging
import logging.handlers
import matplotlib.pyplot as plt


class LibCluster (object):
  def __init__(self, num, edge_source="./results", template_dict="datadict.json"):  
     self.num = num  
     self.edge_source = edge_source
     self.node_dict = template_dict

  def create_network(self):
    self.G = nx.Graph()
    print("network0")
    print(self.G) 
#    G.add_nodes_from([j for j in range(1,i)])
    self.G.add_nodes_from(set(range(self.num)))
    print("network1")
    print(self.G.nodes()) 

  def analyze_network(self, G):
    print("DENDROGRAM0") 
    clusterer = GreedyAgglomerativeClusterer()
    print("DENDROGRAM01") 
    log_dendrogram = clusterer.cluster(self.G)
    print("DENDROGRAM1") 
    print(log_dendrogram.clusters(1))
    print(log_dendrogram.clusters(2))
    print(log_dendrogram.clusters(3))
    print("DENDROGRAM1") 
    print(log_dendrogram.clusters(40))
    print("DENDROGRAM2")

  def build_edges(self): 
    f = open(self.node_dict)
    mydict = json.load(f)
    print(mydict)
    for i in range(2, self.num):
      df1 = pd.read_csv(self.edge_source + '/' + str(i), delimiter=',')
      for j in range(2, self.num):
         if str(i) in mydict.keys() and str(j) in mydict.keys():
            df2 = pd.read_csv(self.edge_source + '/' + str(j), delimiter=',')
            del df2['time'] 
            df_join = pd.concat([df1, df2], axis=1)
            del df_join['time']
            df_join.columns = ['count1', 'count2']
            corr_df = df_join.corr()
            print('---------------------------')
            print(str(i) + ' ' + str(j))
            print(corr_df['count1']['count2'])
            print(mydict[str(i)] + '\n') 
            print(mydict[str(j)] + '\n')
            corr_val = corr_df['count1']['count2']
            if corr_val > 0.8:
               print("Adding " + str(i) + ' ' + str(j))
               self.G.add_edge(i, j, weight=abs(corr_val)*100)
         else:
            print('---------------------------')
            print(str(i) + ' ' + str(j))
            self.G.add_weighted_edges_from(i, j, weight=1)
            print("key does not exist\n") 


  def build_graph_analyze(self):

    self.create_network()
    print("here")
#    print(nx.clustering(self.G))
    self.build_edges()
    print(self.G.nodes())
    print(self.G.edges())
    self.analyze_network(self.G)
    print("here")
    nx.draw(self.G)
    plt.savefig("path.pdf")

if __name__ == '__main__':
  L = LibCluster (394, "./results5", template_dict="template_dict.json")
  L.build_graph_analyze()
