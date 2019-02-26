#!/usr/bin/env python

from __future__ import print_function
import logging
import logging.handlers
import time
import signal
import os
import sys
import argparse
from dateutil import parser as ps
from dateutil import relativedelta
import datetime
import requests
import exceptions
import subprocess
import json
import threading
import pandas as pd
from pandas.tseries.offsets import *

class ShimListener(object):
    def __init__(self, args, ofile):

        self.args = args
        self.outfile = ofile
        self.menv = args.alchemy_env
        self.get_shim_args()
        self.set_alchemy_env()


    def set_alchemy_env(self):

        if self.menv == "stage":
            self.graphite_host = self.graphite_stage_host
            self.space_id = self.stage_space_id
            self.org = self.stage_org
        else: #if env == "prod"
            self.graphite_host = self.graphite_prod_host
            self.space_id = self.prod_space_id
            self.org = self.prod_org
 
        self.auth_token = self.get_credentials()
        self.logmet_auth_token = self.get_credentials_logmet()


    def get_connection_metrics(self, metric):
        try:
           graphite_url = "https://" + self.graphite_host + "/graphite/render"
           metric_path =  self.graphite_space_id +"." +  self.application_id + "." + self.instance_id + "." + self.va_app + "." + metric

           graphite_data =  { "format": "json", "target": metric_path, "from": "-22h", "until": "now"}
           headers = { "X-Auth-Project-Id": self.graphite_space_id, "X-Auth-Token": self.auth_token}
           print(headers)
           print(graphite_data)
           res = requests.get("http://www.google.com")
           res = requests.post(graphite_url, headers=headers, data=graphite_data)
           print(res) 
           if res.ok:
             datapoints_body = json.loads(res.content)
             datapoints = datapoints_body[0]['datapoints']

             valid_datapoints = []
             for datapoint in datapoints:
                if datapoint[0] is not None:
                   valid_datapoints.append(datapoint)
             print(valid_datapoints)
             return(valid_datapoints)

        except Exception:
           time.sleep(60)

    def get_shim_args(self):
        try:
           json_monitor_file = open(self.args.monitors)
           json_creds_file = open(self.args.creds)
           print("done0")
           json_monitor = json.load(json_monitor_file)
           print("done1")
           json_creds = json.load(json_creds_file)
           print("done2")
#           self.outfile = json_monitor["outfile"]
           self.uid =  json_creds["uid"]
           self.upwd =  json_creds["upwd"]
           self.prod_org =  json_creds["prod_org"]
           self.stage_org =  json_creds["stage_org"]

           self.graphite_prod_host = json_creds["graphite_prod_host"]
           self.graphite_stage_host = json_creds["graphite_stage_host"]

           self.stage_space_id = json_creds["stage_space_id"]
           self.prod_space_id = json_creds["prod_space_id"]

           self.graphite_space_id = json_monitor["graphite_space_id"]
           self.application_id = json_monitor["application_id"]
           self.instance_id = json_monitor["instance_id"]
           self.va_app = json_monitor["va_app"]
           self.metric_condition_pairs = json_monitor["metric_condition_pairs"]
           self.metric = json_monitor["metric_condition_pairs"][0][0]
           
           json_monitor_file.close()    
           json_creds_file.close()    

        except Exception:
           print("Error Parsing Arguments")

    def get_credentials_logmet(self):
        '''
        Create curl credentials 
        '''
        try:
           cred_data =  { "user": self.uid, "passwd": self.upwd, "space": self.space_id, "organization": self.org}
           logmet_url = "https://" + "logmet.ng.bluemix.net" + "/login"
           print(cred_data)
           print(logmet_url)
           res = requests.post(logmet_url, data=cred_data)
           
           cred_body = json.loads(res.content)           
           auth_token = cred_body["access_token"]
           self.logmet_space_id = cred_body["space_id"]
           return auth_token
        except Exception:
           print("Get Credentials Failed")

    def csv_to_df(self, csv_file):
      try:
        df = pd.read_csv(csv_file)
        return df     
      except:
        print("Exception in csv file read")   

    def df_to_csv(self, df, csv_file):
      try:
        df.to_csv(csv_file)
      except:
        print("Exception in csv file write")   


    def get_credentials(self):
        '''
        Create curl credentials 
        '''

        try:
           cred_data =  { "user": self.uid, "passwd": self.upwd, "space": self.space_id, "organization": self.org}
           logmet_url = "https://" + self.graphite_host + "/login"
           print(cred_data)
           print(logmet_url)
           res = requests.post(logmet_url, data=cred_data)
           
           cred_body = json.loads(res.content)           
           auth_token = cred_body["access_token"]
           return auth_token
        except Exception:
           print("Get logmet Credentials Failed")

    def toPandasDF(self, datapoints, metric_name):
       [ ax.reverse() for ax in datapoints ]
       df = pd.DataFrame(datapoints, columns=["time", metric_name])
       dfi = df.set_index('time')
       print(metric_name)
       print(dfi)
       return df

    def send_alert(self):
       print("alert: to be implemented") 


    def get_logs_alchemy(self):

        today_str=""
        now = datetime.datetime.now() 
        yest = now - datetime.timedelta(days=1)
        today_str=str(now.year)+"."+str(now.month).zfill(2)+"."+str(now.day-0).zfill(2)
        yest_str=str(yest.year)+"."+str(yest.month).zfill(2)+"."+str(yest.day-0).zfill(2)
        print(today_str)

        if os.path.exists(self.outfile):
            os.remove(self.outfile)
        print(self.space_id)
        print(self.auth_token)
        for i in range(0,20):
           count = i*10000 + 9999
           cmd="curl -v --silent -o " + self.outfile + "_" + str(i) + " -XGET '" + "https://" + "logmet.ng.bluemix.net" + "/elasticsearch/logstash-" + self.logmet_space_id + "-" + today_str + "/_search?size=9999&from=" + str(count) + "&pretty' --header \"X-Auth-Token:" + self.logmet_auth_token + "\"" + " --header \"X-Auth-Project-Id:" + self.logmet_space_id + "\""
           print (cmd) 
           p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
           ret = ""
           for line in p.stdout.readlines():
              ret = ret+line
           print (p) 
           print (ret) 
           time.sleep(10)

    def process_logs_alchemy(self, logs_csv_file="./csv/logs.csv"):
        logfile = open("out1","w")
        logdict = {}
        log_data = []
        f = open(logs_csv_file, "w")
        f.write('time,numlogs\n')
        for i in range(0,20):
           with open(self.outfile + "_" + str(i)) as json_file:
            json_data = json.load(json_file)
            if "hits" in json_data.keys():
                if "hits" in json_data["hits"].keys():
                    log_data.extend(json_data["hits"]["hits"])
        s_start_time = log_data[0]["_source"]["@timestamp"]
        s_end_time = log_data[len(log_data)-1]["_source"]["@timestamp"]
        start_time = ps.parse(s_start_time) 
        end_time = ps.parse(s_end_time) + relativedelta.relativedelta(hours=12)
        time_update1 = start_time
        print (time_update1)
        time_update2 = start_time + relativedelta.relativedelta(minutes=5)   
        resdict = {}
        while time_update2 < end_time:
          print("in time update") 
          print(str(time_update1))
          print(str(time_update2))
          resdict[str(time_update1)] = 0
          resdict[str(time_update2)] = 0 
          for j in range(0,len(log_data)):
              timestamp = log_data[j]["_source"]["@timestamp"]
              dtimestamp = ps.parse(timestamp)

              host = log_data[j]["_source"]["host"]
              logline = log_data[j]["_source"]["message"]

              if host == self.instance_id:
                  if dtimestamp.replace(tzinfo=None) > time_update1.replace(tzinfo=None) and dtimestamp.replace(tzinfo=None) < time_update2.replace(tzinfo=None):
                     resdict[str(time_update1)] = resdict[str(time_update1)] + 1 

                  if dtimestamp.replace(tzinfo=None) > time_update2.replace(tzinfo=None):
                     resdict[str(time_update2)] = 1 

          f.write(str(time_update1) + ',' +  str(resdict[str(time_update1)]) + '\n')
          time_update1 = time_update1 + relativedelta.relativedelta(minutes=5)   
          time_update2 = time_update2 + relativedelta.relativedelta(minutes=5)   
        logfile.close          
        f.close


if __name__ == '__main__':

    try:
        parser = argparse.ArgumentParser(description="")
        parser.add_argument('--infile',  type=str, required=False, help='file name')
        parser.add_argument('--alchemy_env',  type=str, required=True, help='prod:dev')
        parser.add_argument('--monitors',  type=str, required=True, help='metric file')
        parser.add_argument('--creds',  type=str, required=True, help='creds file')
        parser.add_argument('--alert_location',  type=str, required=False, help='location to send alerts')
        parser.add_argument('--sleep_time',  type=str, required=False, help='sleep between metric pulls')
        args = parser.parse_args()

        shim_listener = ShimListener(args, "prod")
        
        shim_listener.get_logs_alchemy()

    except Exception:
        print('Shim Top Level Exception:')



