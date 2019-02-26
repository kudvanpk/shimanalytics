import os
import sys
import numpy as np
import itertools
from gensim import models
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import nltk, string
import re
from dateutil import parser
from dateutil import relativedelta
import dateutil

stemmer = nltk.stem.porter.PorterStemmer()
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)
now = parser.parse("Sat Oct 11 17:13:46 UTC 2003")

class TemplateTableGen(object):
    def __init__(self, args):
       self.code = 0
       self.all_samples = {}
       self.unique_dict = {}
       self.vectorizer = TfidfVectorizer(stop_words='english')
       self.first_date = now 
       self.last_date = now

    def stem_tokens(tokens):
       return [stemmer.stem(item) for item in tokens]

    '''remove punctuation, lowercase, stem'''
    def normalize(text):
       return stem_tokens(nltk.word_tokenize(text.lower().translate(remove_punctuation_map)))

    def cosine_sim(self, text1, text2):

       tfidf = self.vectorizer.fit_transform([text1, text2])
       return ((tfidf * tfidf.T).A)[0,1]

    def preprocess(sentences):
        """
        Some basic text preprocessing, removing line breaks, handling
        punctuation etc.
        """
        punctuation = """.,?!:;(){}[]"""
        sentences = [sent.lower().replace('\n','') for sent in sentences]
        sentences = [sent.replace('<br />', ' ') for sent in sentences]

        #treat punctuation as individual words
        for c in punctuation:
            sentences = [sent.replace(c, ' %s '%c) for sent in sentences]
        sentences = [sent.split() for sent in sentences]
        return sentences

    # "@timestamp":"2016-01-10T00:00:36.812Z"
    def get_tdate(self, sentences1):
        m = re.search("\"@timestamp\":\".*Z\",", sentences1)
        mstr= ''
        if m:
          mstr = m.group(0)
          if len(mstr.split(':\"')) > 0:
             mstr = mstr.split(':\"')[1].replace('\",', '')
        return mstr

    def remove_date(self, sentences1):
        print("in removed") 
        if " " in sentences1: 
         if len(sentences1.split()) > 1:
           s1 = sentences1.split()[0]
           s2 = sentences1.split()[1]
         if '2016' in s1:
           sentences1 = sentences1.replace(s1, '')
           sentences1 = sentences1.replace(s2, '')
        print("done removed") 
        return(sentences1)

    def string_clean(self, sentences1):
        sentences1 = sentences1.replace('"_source"', '')
        sentences1 = sentences1.replace('"message"', '')
        sentences1 = sentences1.replace('"', '')
        sentences1 = sentences1.replace('\'', '')
        sentences1 = sentences1.replace('{', '')
        sentences1 = sentences1.replace('}', '')
        sentences1 = sentences1.replace(':', '')
#        sentences1 = self.remove_date(sentences1)
        return sentences1

    def get_data(self, inputfile):
        """
        Fetching files and separating samples into sentences to compare
        """
        fname = inputfile
        with open(inputfile,'r') as f:
            samples = f.readlines()
        for sample in samples:
            if "timestamp" in sample:
               self.first_date = parser.parse(self.get_tdate(sample))
               print self.first_date
               break

        for sample in reversed(samples):
            if "timestamp" in sample:
               self.last_date = parser.parse(self.get_tdate(sample))
               print self.last_date
               break
        print self.first_date          
        print self.last_date          
        print self.last_date - self.first_date
        print self.first_date + relativedelta.relativedelta(minutes=10)
        sample = samples[0]
        f = open('resfile4', 'w')
        self.all_samples = {}
        self.process_samples(sample, samples[1:], f)
        with open(template_file, 'w') as fp:
           json.dump(self.unique_dict, fp)
        f.close()
        print("HERE2")
        return self.all_samples

    def process_samples(self, xsample, xsamples, f, clean=True):

        f.write('==================================================\n')

        res_sample = []
        if "_source" in xsample:   
           sentences1 = xsample
           sentences1 = self.string_clean(sentences1)
           self.code = self.code + 1
           tdate = self.get_tdate(sentences1)
           print("HERE1")
           if clean is True: 
               self.all_samples.setdefault(str(self.code), []).append(tdate) 
           self.unique_dict[str(self.code)] = sentences1         
           f.write(sentences1) 
           f.write('\n') 
           f.write('---------------matches--------------\n') 
           for tsample in xsamples:
               sentences2 = tsample
               if "_source" in sentences2:  
                 print("HERE9")
                 tdate = self.get_tdate(sentences2)
                 print("HERE10")
                 print(sentences2)
                 sentences2 = self.string_clean(sentences2)
                 print("HERE11")
                 res = self.cosine_sim(sentences1, sentences2)
                 print("HERE12")
                 res15 = 0.0
                 if len(sentences1) > 30 and len(sentences2) > 30:
                    res15 = self.cosine_sim(sentences1[:29], sentences2[:29])
                 if res > 0.7 or res15 > 0.85:
                    f.write(sentences2) 
                    f.write('\n') 
                    f.write(str(res))
                    f.write('\n') 
                    f.write('----------------------------------------\n') 
                    self.all_samples.setdefault(str(self.code), []).append(tdate) 
                 else:
                    res_sample.append(tsample)     
           if len(res_sample) > 1:
                print("HERE4")
                self.process_samples(res_sample[0],res_sample[1:], f)
        else:
           if len(xsamples) > 1:
                print("HERE5")
                self.process_samples(xsamples[0],xsamples[1:], f)

    def labelize_text(sentences, label_type):
        """
        A special requirement for gensim doc2vec,
        each sentence has to be labeled and turned
        into LabeledSentence object.
        """
        labelized = []
        LabeledSentence = models.doc2vec.LabeledSentence

        for ind,sent in enumerate(sentences):
            label = '%s_%s'%(label_type,ind)
            labelized.append(LabeledSentence(sent, [label]))
        return labelized


    def PCA_model(samples):
        """
        Alternative to Doc2Vec for data vectorization
        """
        vectorizer = TfidfVectorizer(stop_words='english')
        svd = TruncatedSVD(n_components=5, random_state=42)
        pca = make_pipeline(vectorizer, svd, Normalizer(copy=False))
        model = pca.fit(samples)
        return model


    def D2V_model(sentences1, sentences2):
        """
        Initializing and training a Doc2Vec model
        """
        corpus = sentences1+sentences2
        d2v = models.Doc2Vec(min_count=1, window=10, size=10, sample=1e-3, 
                                                        negative=5, workers=1)
        d2v.build_vocab(corpus)
        d2v.train(corpus)
        return d2v


    def get_vecs(model, sentences, size):
        """
        Vectorizes input sentences using pre-trained Doc2Vec model
        and returns as numpy arrays.
        """
        vecs = [np.array(model[sent.labels[0]]).reshape((1, size)) for sent in sentences]
        return np.concatenate(vecs)


    def log_templates_and_tables(self, log_file, template_file='/opt/datadict.json', tables_dir='/opt/results'):
       print(log_file)     
       num_nodes = self.log_templates(log_file, template_file)
       self.log_tables_for_clustering(template_file, tables_dir)
       return num_nodes

    def log_templates(self, log_file, template_file='datadict.json'):
        print("log file")     
        self.all_samples = self.get_data(log_file) 
        print self.unique_dict
        with open(template_file, 'w') as fp:
           json.dump(self.unique_dict, fp)
        return len(self.unique_dict) 

    def log_tables_for_clustering(self, template_file='datadict.json', tables_dir='./results'):

        for as_key, as_val in self.all_samples.iteritems():
            f = open('tables_dir'+ as_key, 'w')   
            f.write('time' + ',' +  'count' + '\n')
            as_val.sort()
            time_update1 = self.first_date    
            time_update2 = self.first_date + relativedelta.relativedelta(minutes=10)   
            while time_update2 < self.last_date:
               res = [ val for val in as_val if parser.parse(val).replace(tzinfo=None) > time_update1.replace(tzinfo=None) and parser.parse(val).replace(tzinfo=None) < time_update2.replace(tzinfo=None) ]
               f.write(str(time_update1) + ',' +  str(len(res)) + '\n')
               time_update1 = time_update1 + relativedelta.relativedelta(minutes=10)       
               time_update2 = time_update2 + relativedelta.relativedelta(minutes=10)       
            f.close()

if __name__ == '__main__':
    log_file = sys.argv[1]
    log_templates_and_tables(log_file)


