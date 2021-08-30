#!/users/jflournoy/.conda/envs/abcd_ml_3.7/bin/python
print('importing libs...')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder 
import argparse
from joblib import dump
import logging

print('setting up parser')
# Models: Age + Sex in all
#   *Day level: 1-7, 1-14, 1-21
#   *Week level: 1, 1-2, 1-3
#   *Summary
#   Time-since CBCL interaction 
#   Fitbit variables: Most common versus everything
#     Step count, HR (resting), Sleep duration
#   Physical activity versus PA + Sleep in the subsample with sleep
#     Compare this to full sample with sleep data imputed.
#   Benchmarking
#     Demographics (see variables sheet)
#       interview_age
#       Sex
#       demo_race
#     Fitbit minimal: Age + Sex
#     Parental history?


parser = argparse.ArgumentParser(description='Run some ML models on ABCD fitbit data')
parser.add_argument('-p', '--predictorset', metavar='pset', type=str, 
                    help='A string specifying which predictor set to use. See code for specifics. \
                    Some sets require you to also specify the summary level (-s) and a time subset (-t).')
parser.add_argument('-s', '--summary', metavar='level', type=str, 
                    help='A string specifying which level, [daily|weekly|id], of summary to use.')
parser.add_argument('-t', '--time', metavar='max_time', type=int, 
                    help='An integer specifying the maximum time, either days, or weeks, you want to use. \
                    -t 7 would specify 7 days if this was a day-data model. Should be 7, 14, or 21 for \
                    days, or 1, 2, or 3 for weeks.')
parser.add_argument('-y', '--outcome', metavar='outcome', type=str,
                   help='A string specifying the outcome level: [subscale|scale|overall].')
parser.add_argument('-ni', '--n_inner', metavar='N', type=int,
                   help='Number of splits for inner-loop CV.')
parser.add_argument('-no', '--n_outer', metavar='N', type=int,
                   help='Number of splits for outer-loop CV.')
parser.add_argument('-c', '--cores', metavar='Cores', type=int,
                   help='Number of cores available.')
parser.add_argument('--slurmid', metavar='ID', type=str,
                   help='Slurm id, for logging.')

print('parsing args')
#args = parser.parse_args(['-p', 'pa', '-s', 'id', '-t', '2', '-y', 'overall', '-ni', '5', '-no', '2', '-c' , '1'])
args = parser.parse_args()

logging.basicConfig(filename='log/abcd-ml_{}.log'.format(args.slurmid), level=logging.DEBUG)

logging.info(args)

logging.info('reading data')
train_data = pd.read_csv('train_data.csv')
train_data['sex'] = train_data['sex'].astype('category').cat.codes
# y is 6-13, or 14-15, or 16
#  PA daily: 20-46
#  PA weekly: 47-64
#  PA summary: 65-82
#
#  sleep daily: 84-125
#  sleep weekly: 126-153
#  sleep summary: 154-181
#  Group index: 182
# for thing in zip(range(len(train_data.columns)), train_data.columns):
#     print(str(thing[0]) + ': ' + thing[1])

pvarranges = {"pa" : {"daily" : range(20,47),
                     "weekly" : range(47,65),
                     "id" : range(65,82)},
             "sleep" : {"daily" : range(84,126),
                        "weekly" : range(126,154),
                        "id" : range(154,181)}
             }
yvarranges = {"subscale" : range(6, 14),
              "scale" : range(14,16),
              "overall" : 16}

### problem when pocolrange is a single column, list()ing it doesn't make a list
### Sex column is not numeric.

pcolrange = pvarranges[args.predictorset][args.summary]
ycolrange = yvarranges[args.outcome]
pcolnames = list(train_data.columns[pcolrange]) + ['sex', 'interview_age']
if type(ycolrange) is int:
    ycolnames = [train_data.columns[ycolrange]]
else:
    ycolnames = list(train_data.columns[ycolrange])

for thing in zip(range(len(pcolnames)), list(pcolnames)):
    logging.info(str(thing[0]) + ': ' + thing[1])
for thing in zip(range(len(ycolnames)), list(ycolnames)):
    logging.info(str(thing[0]) + ': ' + thing[1])

logging.debug("args.summary is {} of type {} and truth value is {}".format(args.summary, type(args.summary), args.summary == "id"))
    
if args.predictorset in ["pa", "sleep"]:
    if args.summary in ["daily", "weekly"]:
        timecol = "daynum" if args.summary == "daily" else "weekno"
        timerange = range(0, args.time)
        timeindex = train_data[timecol].isin(timerange)
        these_train_data = train_data[pcolnames + ycolnames + [timecol, 'idnum']][timeindex].drop_duplicates()
        model_suffix="_{}_{}".format(args.summary, args.time)
        logging.info('Time column is {}, and time range is {}'.format(timecol, list(timerange)))
    elif args.summary == "id":
        logging.debug("colnames to select are {}, {}, {}".format(pcolnames, ycolnames, ['idnum']))
        logging.debug("shape of selection will be {}.".format(train_data[pcolnames + ycolnames + ['idnum']].drop_duplicates().shape))
        these_train_data = train_data[pcolnames + ycolnames + ['idnum']].drop_duplicates()
        model_suffix="_{}".format(args.summary)
    X = these_train_data[pcolnames].to_numpy()
    Y = these_train_data[ycolnames].to_numpy()
    groups = these_train_data['idnum'].to_numpy()
else:
    logging.error("Other models not yet specified.")
    

outname="out/abcd-ml_{}{}".format(args.predictorset, model_suffix)
logging.info("outfile is {}".format(outname))

N_outer=args.n_outer
N_inner=args.n_outer

test_size=.2

cvsplitter_outer = GroupShuffleSplit(n_splits=N_outer, test_size=test_size)
cvsplitter_inner = GroupShuffleSplit(n_splits=N_inner, test_size=test_size) 
imputer = SimpleImputer(missing_values=np.nan, add_indicator=True)

logging.info("Starting outer CV, N = {}".format(N_outer))

#Outer loop over N splits
split_index = 0
for train_idx, test_idx in cvsplitter_outer.split(X, Y, groups):
    groups_train = groups[train_idx]
    X_train = X[train_idx]
    Y_train = Y[train_idx]

    groups_test = groups[test_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    
    regressor=MultiTaskElasticNetCV(l1_ratio = [.1, .5, .7, .9, .95, .99, 1], 
                                    n_jobs = args.cores, 
                                    cv = list(cvsplitter_inner.split(X_train, Y_train, groups_train)))
    estimator = make_pipeline(imputer, regressor)
    logging.info("Training...")
    estimator.fit(X_train, Y_train)
    
    logging.info('Training: {:1.3} Testing: {:1.3}'.format(estimator.score(X_train, Y_train), estimator.score(X_test, Y_test)))
    
    out_dict={"score_train" : estimator.score(X_train, Y_train),
              "score_test" : estimator.score(X_test, Y_test),
              "intercept" : estimator.named_steps['multitaskelasticnetcv'].intercept_ ,
              "coef" : estimator.named_steps['multitaskelasticnetcv'].coef_ ,
              "alpha" : estimator.named_steps['multitaskelasticnetcv'].alpha_ ,
              "alphas" : estimator.named_steps['multitaskelasticnetcv'].alphas_ ,
              "mse_path" : estimator.named_steps['multitaskelasticnetcv'].mse_path_ ,
              "l1_ratio" : estimator.named_steps['multitaskelasticnetcv'].l1_ratio_ ,
              "n_iter" : estimator.named_steps['multitaskelasticnetcv'].n_iter_,
              "score_train" : estimator.score(X_train, Y_train),
              "score_test" : estimator.score(X_test, Y_test),
              "X_train" : X_train,
              "Y_train" : Y_train,
              "X_test" : X_test,
              "Y_test" : Y_test,
              "Y_pred_train" : estimator.predict(X_train),
              "Y_pred_test" : estimator.predict(X_test),
              "estimator" : estimator}
    logging.info("Pickling out_dict")
    dump(out_dict,'{}_s{:03}.pkl'.format(outname, split_index))
    split_index += 1
print("Done!")