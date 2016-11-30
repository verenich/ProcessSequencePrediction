# imports 
from __future__ import division
from keras.models import load_model
import csv
import copy
import numpy as np
import distance
from itertools import izip
import jellyfish
import unicodecsv
from sklearn import metrics
from math import sqrt
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter

csvfile = open('../data/helpdesk_full.csv', 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
ascii_offset = 161

lastcase = ''
line = ''
firstLine = True
lines = []
timeseqs = []
timeseqs2 = []
times = []
times2 = []
numlines = 0
casestarttime = None
lasteventtime = None
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
        line = ''
        times = []
        times2 = []
        numlines+=1
    line+=unichr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
numlines+=1

divisor = np.mean([item for sublist in timeseqs for item in sublist])
print('divisor: {}'.format(divisor))
divisor2 = np.mean([item for sublist in timeseqs2 for item in sublist])
print('divisor2: {}'.format(divisor2))
divisor3 = np.mean(map(lambda x: np.mean(map(lambda y: x[len(x)-1]-y, x)), timeseqs2))
print('divisor3: {}'.format(divisor3))

elems_per_fold = int(round(numlines/3))
fold1 = lines[:elems_per_fold]
fold1_t = timeseqs[:elems_per_fold]
fold1_t2 = timeseqs2[:elems_per_fold]

fold2 = lines[elems_per_fold:2*elems_per_fold]
fold2_t = timeseqs[elems_per_fold:2*elems_per_fold]
fold2_t2 = timeseqs2[elems_per_fold:2*elems_per_fold]

lines = fold1 + fold2
lines_t = fold1_t + fold2_t
lines_t2 = fold1_t2 + fold2_t2

step = 1
sentences = []
softness = 0
next_chars = []
#lines = text.splitlines()
#lines = map(lambda x: '{'+x+'}',lines)
lines = map(lambda x: x+'!',lines)
maxlen = max(map(lambda x: len(x),lines))

chars = map(lambda x : set(x),lines)
chars = list(set().union(*chars))
chars.sort()
target_chars = copy.copy(chars)
chars.remove('!')
#target_chars.remove('{')
print('total chars: {}, target chars: {}'.format(len(chars), len(target_chars)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
target_char_indices = dict((c, i) for i, c in enumerate(target_chars))
target_indices_char = dict((i, c) for i, c in enumerate(target_chars))
#indices_char[len(target_chars)] = '{'
print(indices_char)


lastcase = ''
line = ''
firstLine = True
lines = []
timeseqs = []  # relative time since previous event
timeseqs2 = [] # relative time since case start
timeseqs3 = [] # absolute time of previous event
times = []
times2 = []
times3 = []
numlines = 0
casestarttime = None
lasteventtime = None
csvfile = open('../data/helpdesk_full.csv', 'r')
spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
next(spamreader, None)  # skip the headers
for row in spamreader:
    t = time.strptime(row[2], "%Y-%m-%d %H:%M:%S")
    if row[0]!=lastcase:
        casestarttime = t
        lasteventtime = t
        lastcase = row[0]
        if not firstLine:        
            lines.append(line)
            timeseqs.append(times)
            timeseqs2.append(times2)
            timeseqs3.append(times3)
        line = ''
        times = []
        times2 = []
        times3 = []
        numlines+=1
    line+=unichr(int(row[1])+ascii_offset)
    timesincelastevent = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(lasteventtime))
    timesincecasestart = datetime.fromtimestamp(time.mktime(t))-datetime.fromtimestamp(time.mktime(casestarttime))
    midnight = datetime.fromtimestamp(time.mktime(t)).replace(hour=0, minute=0, second=0, microsecond=0)
    timesincemidnight = datetime.fromtimestamp(time.mktime(t))-midnight
    timediff = 86400 * timesincelastevent.days + timesincelastevent.seconds
    timediff2 = 86400 * timesincecasestart.days + timesincecasestart.seconds
    times.append(timediff)
    times2.append(timediff2)
    times3.append(datetime.fromtimestamp(time.mktime(t)))
    lasteventtime = t
    firstLine = False

# add last case
lines.append(line)
timeseqs.append(times)
timeseqs2.append(times2)
timeseqs3.append(times3)
numlines+=1

fold3 = lines[2*elems_per_fold:]
fold3_t = timeseqs[2*elems_per_fold:]
fold3_t2 = timeseqs2[2*elems_per_fold:]
fold3_t3 = timeseqs3[2*elems_per_fold:]
#fold3_t4 = timeseqs4[2*elems_per_fold:]

lines = fold3
lines_t = fold3_t
lines_t2 = fold3_t2
lines_t3 = fold3_t3
#lines_t4 = fold1_t4 + fold2_t4

# set parameters
predict_size = 1

# load model
model = load_model('output_files/models/no_parikh/model_33-1.51.h5')

# define helper functions
def encode(sentence, times, times3, maxlen=maxlen):
    num_features = len(chars)+5
    X = np.zeros((1, maxlen, num_features), dtype=np.float32)
    leftpad = maxlen-len(sentence)
    times2 = np.cumsum(times)
    for t, char in enumerate(sentence):
        midnight = times3[t].replace(hour=0, minute=0, second=0, microsecond=0)
        timesincemidnight = times3[t]-midnight
        multiset_abstraction = Counter(sentence[:t+1])
        for c in chars:
            if c==char:
                X[0, t+leftpad, char_indices[c]] = 1
        X[0, t+leftpad, len(chars)] = t+1
        X[0, t+leftpad, len(chars)+1] = times[t]/divisor
        X[0, t+leftpad, len(chars)+2] = times2[t]/divisor2
        X[0, t+leftpad, len(chars)+3] = timesincemidnight.seconds/86400
        X[0, t+leftpad, len(chars)+4] = times3[t].weekday()/7
    #print(X)
    return X

def getSymbol(predictions):
    maxPrediction = 0
    symbol = ''
    i = 0;
    for prediction in predictions:
        if(prediction>=maxPrediction):
            maxPrediction = prediction
            symbol = target_indices_char[i]
        i += 1
    return symbol

one_ahead_gt = []
one_ahead_pred = []

two_ahead_gt = []
two_ahead_pred = []

three_ahead_gt = []
three_ahead_pred = []


# make predictions
with open('output_files/folds/fold3_preds.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(["Prefix length", "Groud truth", "Predicted", "Levenshtein", "Damerau", "Jaccard", "Ground truth times", "Predicted times", "RMSE", "MAE", "Median AE"])
    for prefix_size in range(2,maxlen):
        print(prefix_size)
        for line, times, times2, times3 in izip(lines, lines_t, lines_t2, lines_t3):
            times.append(0)
            cropped_line = ''.join(line[:prefix_size])
            cropped_times = times[:prefix_size]
            cropped_times3 = times3[:prefix_size]
            if len(times2)<prefix_size:
                continue # make no prediction for this case, since this case has ended already
            ground_truth = ''.join(line[prefix_size:prefix_size+predict_size])
            #print(map(lambda x: x.encode('utf-8'), line))
            #print(map(lambda x: x.encode('utf-8'), cropped_line))
            #print(times2)
            ground_truth_t = times2[prefix_size-1]
            case_end_time = times2[len(times2)-1]
            ground_truth_t = case_end_time-ground_truth_t
            #print('times: {}'.format(times))
            #print('gt: {}'.format(ground_truth_t))
            predicted = ''
            #print(ground_truth_t)
            #print(predicted_t)
            #print(cropped_times)
            #print('curr len: {}'.format(len(cropped_line)))
            #print('line len: {}'.format(len(line)))
            total_predicted_time = 0
            for i in range(predict_size):
                enc = encode(cropped_line, cropped_times, cropped_times3)
                #print(ground_truth_t[i])
                y = model.predict(enc, verbose=0)
                #print(y)
                #print('y:      {}'.format(y))
                y_char = y[0][0]
                #print('y_char: {}'.format(y_char))
                y_t = y[1][0][0]
                #print('y_t: {}'.format(y_t))
                prediction = getSymbol(y_char)
                
                #prediction = ground_truth[i]
                #y_t = ground_truth_t[i]/divisor
              
                cropped_line += prediction
                if y_t<0:
                    y_t=0
                cropped_times.append(y_t)
                
                # add y_t seconds to cropped_times3
                if prediction == '!': # end of case was just predicted, therefore, stop predicting further into the future
                    one_ahead_pred.append(total_predicted_time)
                    one_ahead_gt.append(ground_truth_t)
                    print('! predicted, end case')
                    break
                y_t = y_t * divisor3
                cropped_times3.append(cropped_times3[-1] + timedelta(seconds=y_t))
                total_predicted_time = total_predicted_time + y_t
                predicted += prediction
            output = []
            if len(ground_truth)>0:
                output.append(prefix_size)
                output.append(unicode(ground_truth).encode("utf-8"))
                output.append(unicode(predicted).encode("utf-8"))
                output.append(1 - distance.nlevenshtein(predicted, ground_truth))
                output.append(1 - (jellyfish.damerau_levenshtein_distance(unicode(predicted), unicode(ground_truth)) / max(len(predicted),len(ground_truth))))
                output.append(1 - distance.jaccard(predicted, ground_truth))
                output.append(ground_truth_t)
                output.append(total_predicted_time)
                output.append('')
                output.append(metrics.mean_absolute_error([ground_truth_t], [total_predicted_time]))
                output.append(metrics.median_absolute_error([ground_truth_t], [total_predicted_time]))
                spamwriter.writerow(output)

fig = plt.figure()        
ax1 = fig.add_subplot(311)
#ax1.title.set_text('predictions 1 event ahead')
#ax1.set_xlabel('ground truth time since last event')
#ax1.set_ylabel('predicted time since last event')
ax1 = plt.scatter(one_ahead_gt, one_ahead_pred)
#ax1.plot([0, 0], [600000, 600000], 'k-')
ax2 = fig.add_subplot(312)
ax2 = plt.scatter(two_ahead_gt, two_ahead_pred)
#ax2.title.set_text('predictions 2 events ahead')
#ax2.set_xlabel('ground truth time since last event')
#ax2.set_ylabel('predicted time since last event')
#ax2.plot([0, 0], [600000, 600000], 'k-')
ax3 = fig.add_subplot(313)
ax3 = plt.scatter(three_ahead_gt, three_ahead_pred)
#ax3.title.set_text('predictions 3 events ahead')
#ax3.set_xlabel('ground truth time since last event')
#ax3.set_ylabel('predicted time since last event')
#ax3.plot([0, 0], [600000, 600000], 'k-')
plt.show()