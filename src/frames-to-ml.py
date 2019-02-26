import os
import csv
import re
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import skflow
import random

path = './frames'
paramlist = ['fsshdexist', 'foptsbinssh', 'flocalbinsshd', 'frootsshd', 'fbinstart', 'fbinbash', 'fbinsuperd', 'fconfsuperd', 'flconfsuperdsshd', 'flconfsuperdoptsshd','flconfsuperdlocalssh', 'flconfrootsshd', 'fstartsshd-D', 'popenssh-server', 'popenSSH', 'deport', 'deesbinsshd', 'deeoptsbinsshd', 'deelocalsshd', 'deesshd', 'deestartsshd', 'decoptsbinssh','declocalbinssh', 'decsshd', 'decstartssh', 'filename', 'tag']

def create_frames (path):
  framelist = []
  for filename in os.listdir(path):
     valdict = {}
     for param in paramlist:
        valdict[param] = 0
     with open(path + "/" + filename) as f: 
#         print(filename)    
         valdict['filename'] = filename 
         for line in f:
            if line.startswith("file"):
               if '/usr/sbin/sshd' in line:
                   valdict['fsshdexist'] = 1     
               if '/opt/bin/sshd' in line:
                   valdict['foptsbinssh'] = 1     
               if '/usr/local/bin/sshd' in line:
                   valdict['flocalbinsshd'] = 1     
               if ' \"/sshd' in line:
                   valdict['frootsshd'] = 1     
               if '/usr/bin/start.sh' in line:
                   valdict['fbinstart'] = 1     
               if '/bin/bash' in line:
                   valdict['fbinbash'] = 1     
               if '/usr/bin/supervisord' in line:
                   valdict['fbinsuperd'] = 1     
               if '/etc/supervisor/conf.d/supervisord.conf' in line:
                  valdict['fconfsuperd'] = 1     
                  if 'command=/usr/sbin/sshd' in line:
                    valdict['flconfsuperdsshd'] = 1
                  if 'command=/opt/sbin/sshd' in line:
                    valdict['flconfsuperdoptsshd'] = 1
                  if 'command=/usr/local/sbin/sshd' in line:
                    valdict['flconfsuperdlocalssh'] = 1
                  if 'command=/sshd' in line:
                    valdict['flconfrootsshd'] = 1
               if '/usr/bin/start.sh' in line:
                  if '/usr/local/sbin/sshd -D' in line:
                    valdict['fstartsshd-D'] = 1

            if line.startswith("package"):
               if 'openssh-server' in line:
                    valdict['popenssh-server'] = 1
               if 'openSSH' in line:
                    valdict['popenssh'] = 1
            if line.startswith("dockerinspect"):
               if '\"Entrypoint\":\"/usr/sbin/sshd\"' in line:
                    valdict['deesbinsshd'] = 1
               if '\"Entrypoint\":\"/opt/sbin/sshd\"' in line:
                    valdict['deeoptsbinsshd'] = 1
               if '\"Entrypoint\":\"/usr/local/bin/sshd\"' in line:
                    valdict['deelocalsshd'] = 1
               if '\"Entrypoint\":\"sshd\"' in line:
                    valdict['deesshd'] = 1
               if '\"Entrypoint\":/usr/bin/start.sh\"' in line:
                    valdict['deestartsshd'] = 1
               if '\"Cmd\":\"/usr/sbin/sshd\"' in line:
                    valdict['decsbinsshd'] = 1
               if '\"Cmd\":\"/opt/sbin/ssh\"' in line:
                    valdict['decoptsbinssh'] = 1
               if '\"Cmd\":\"/usr/local/bin/sshd\"' in line:
                    valdict['declocalbinssh'] = 1
               if '\"Cmd\":\"sshd\"' in line:
                    valdict['decsshd'] = 1
               if '\"Cmd\":/usr/bin/start.sh\"' in line:
                    valdict['decstartssh'] = 1
               if '\"ExposedPorts\":\(d+)' in line:
                  match = re.match( r'\"ExposedPorts\":\(d+)', line)
                  valdict['deport'] = match.group(1) 
         with open('./sshd-tags-full') as f: 
            for line in f:    
               match = line.split()
               if match[3] in filename: 
                  if match[1] == 'not-listening':
                    valdict['tag'] = 0
                  else: 
                    valdict['tag'] = 1
         framelist.append(valdict)
  print(valdict)  
  df = pd.DataFrame.from_records(framelist)
  df.set_index('filename', inplace=True)
  df.reset_index(drop=True, inplace = True)
  print(df)
  return df

def analyze_df(df):
   paramlist.remove('filename')
   paramlist.remove('tag')
   print paramlist
   FEATURES = df[paramlist]
   TARGETS = df[["tag"]]
   print FEATURES
   print TARGETS
   splits = cv.train_test_split(FEATURES, TARGETS, test_size=0.8)
   print "splits"
   print splits
   X_train, X_test, y_train, y_test = splits


   model = LogisticRegression()
   model.fit(X_train, y_train.values.ravel())

#   model = skflow.TensorFlowLinearClassifier(n_classes=2, batch_size=128, steps=500, learning_rate=0.05)
#   model.fit(X_train, y_train)


   expected = y_test
   predicted = model.predict(X_test)

   print "Logistic Regression Classifier \n ssh dataset"
#   print "Tensor Flow Classifier \n ssh dataset"
   print classification_report(expected, predicted)


def write_csv (framelist): 
   keys = framelist[0].keys()
   with open('data.csv', 'wb') as output_file:
    dict_writer = csv.DictWriter(output_file, keys)
    dict_writer.writeheader()
    dict_writer.writerows(framelist)


if __name__ == '__main__':
  sshdf = create_frames(path)  
  analyze_df(sshdf)
