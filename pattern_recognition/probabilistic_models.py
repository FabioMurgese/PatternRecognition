#!/usr/bin/env python3
from hmmlearn.hmm import GaussianHMM
import os
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator
from datetime import datetime
import numpy as np
import collections

results_path = '../results/HMM/'

energy_data = os.path.abspath('../datasets/energy_data/energydata_complete.csv')
start_date = datetime.strptime('2016-01-11', '%Y-%m-%d')
end_date = datetime.strptime('2016-03-11', '%Y-%m-%d')
start_date_sample = datetime.strptime('2016-03-12', '%Y-%m-%d')
end_date_sample = datetime.strptime('2016-05-27', '%Y-%m-%d')
K_MIN = 2
K_MAX = 10
N = 100  # number of samples to generate


# take energy consumption of appliances
appliances = pd.read_csv(energy_data, header=0, index_col=0, parse_dates=True, usecols=[0, 1])
appliances_TR = appliances.loc[start_date:end_date]

#n_examples = len(appliances)
#TR_SIZE = int(n_examples * 80 / 100)
#appliances_TR = appliances[0:TR_SIZE]
#appliances_TS = appliances[TR_SIZE + 1:len(appliances)]

# initialize the score dict for appliances
a_scores = collections.defaultdict(list)

for k in range(K_MIN, K_MAX):
    # Create the HMM and fit it to data
    appliances_model = GaussianHMM(n_components=k, covariance_type="diag", n_iter=1000).fit(appliances)
    # Decode the optimal sequence of internal hidden state (Viterbi)
    # The internal state of the model only determines the probability distribution of the observed variables
    a_hidden_states = appliances_model.predict(appliances_TR)

    appliances.plot(kind='kde')
    plt.title("Density distribution")
    plt.savefig(results_path + 'appliances_distr.png')
    plt.close()

    # Transition matrix (size (M x M) where M is number of hidden states)
    print("Transition matrix")
    print(appliances_model.transmat_)

    print("Means and vars of each hidden state")
    for i in range(appliances_model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", appliances_model.means_[i])
        print("var = ", np.diag(appliances_model.covars_[i]))

    fig, axs = plt.subplots(appliances_model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, appliances_model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = a_hidden_states == i
        ax.plot(appliances_TR[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))
        # Format the ticks.
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    plt.plot()
    plt.savefig(results_path + 'appliances_hidden_states_' + str(appliances_model.n_components) + '.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(appliances_TR)
    ax = list(range(0, len(appliances_TR)))
    colours = cm.rainbow(np.linspace(0, 1, appliances_model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = a_hidden_states == i
        plt.plot(appliances_TR[mask], ".", c=colour, label="{0}th hidden state".format(i))
    plt.title("Hidden States")
    plt.legend(loc='best')
    plt.plot()
    plt.savefig(results_path + 'appliances_hidden_states_aggregate_' + str(appliances_model.n_components) + '.png')
    plt.close()

    # Generate samples(visible, hidden)
    X_a, Z_a = appliances_model.sample(N)
    appliances_TS = appliances.loc[start_date_sample:end_date_sample]

    # Compare the N sampled elements wrt ground truth data (appliances)
    plt.plot(X_a, label='predictions')
    plt.plot(appliances_TS.Appliances.to_numpy()[0:N], label='real values')
    plt.xlabel("samples")
    plt.ylabel("consumptions")
    plt.legend(loc='best')
    plt.title("Appliances")
    plt.savefig(results_path + 'appliances_predictions_' + str(appliances_model.n_components) + '.png')
    plt.close()

    print("Score of sampled appliances consumptions: ", str(appliances_model.score(X_a)))

    # compare the different models thanks to the log likelihood
    a_scores[appliances_model.n_components].append(appliances_model.score(X_a))
    lists = sorted(a_scores.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.xlabel("number of hidden states")
    plt.ylabel("log likelihood")
    plt.title("Appliances models scores")
    plt.savefig(results_path + 'appliances_scores.png')
    plt.close()

# take energy consumption of lights
lights = pd.read_csv(energy_data, header=0, index_col=0, parse_dates=True, usecols=[0, 2])
lights_TR = lights.loc[start_date:end_date]

#lights_TR = lights[0:TR_SIZE]
#lights_TS = lights[TR_SIZE + 1:len(lights)]

# initialize the score dict for lights
l_scores = collections.defaultdict(list)

# plot the density distribution of lights consumptions
lights.plot(kind='kde')
plt.title("Density distribution")
plt.savefig(results_path + 'lights_distr.png')
plt.close()

for k in range(K_MIN, K_MAX):
    # Create the HMM and fit it to data
    lights_model = GaussianHMM(n_components=k, covariance_type="diag", n_iter=1000).fit(lights)
    # Decode the optimal sequence of internal hidden state (Viterbi)
    l_hidden_states = lights_model.predict(lights_TR)

    # Transition matrix (size (M x M) where M is number of hidden states)
    print("Transition matrix")
    print(lights_model.transmat_)

    print("Means and vars of each hidden state")
    for i in range(lights_model.n_components):
        print("{0}th hidden state".format(i))
        print("mean = ", lights_model.means_[i])
        print("var = ", np.diag(lights_model.covars_[i]))

    fig, axs = plt.subplots(lights_model.n_components, sharex=True, sharey=True)
    colours = cm.rainbow(np.linspace(0, 1, lights_model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        mask = l_hidden_states == i
        ax.plot(lights_TR[mask], ".-", c=colour)
        ax.set_title("{0}th hidden state".format(i))
        # Format the ticks.
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_minor_locator(MonthLocator(bymonthday=15))
    plt.plot()
    plt.savefig(results_path + 'lights_hidden_states_' + str(lights_model.n_components) + '.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(lights_TR)
    ax = list(range(0, len(lights_TR)))
    colours = cm.rainbow(np.linspace(0, 1, lights_model.n_components))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        mask = l_hidden_states == i
        plt.plot(lights_TR[mask], ".", c=colour, label="{0}th hidden state".format(i))
    plt.title("Hidden States")
    plt.legend(loc='best')
    plt.plot()
    plt.savefig(results_path + 'lights_hidden_states_aggregate_' + str(lights_model.n_components) + '.png')
    plt.close()

    # Generate samples(visible, hidden)
    X_l, Z_l = lights_model.sample(N)
    lights_TS = lights.loc[start_date_sample:end_date_sample].lights.to_numpy()

    # Compare the 100 sampled elements wrt ground truth data (lights)
    plt.plot(X_l, label='predictions')
    plt.plot(lights_TS[0:N], label='real values')
    plt.xlabel("samples")
    plt.ylabel("consumptions")
    plt.legend(loc='best')
    plt.title("Lights")
    plt.savefig(results_path + 'lights_predictions_' + str(lights_model.n_components) + '.png')
    plt.close()

    # compare the different models thanks to the log likelihood
    l_scores[lights_model.n_components].append(lights_model.score(X_l))
    lists = sorted(l_scores.items())  # sorted by key, return a list of tuples
    x, y = zip(*lists)  # unpack a list of pairs into two tuples
    plt.plot(x, y)
    plt.xlabel("number of hidden states")
    plt.ylabel("log likelihood")
    plt.title("Lights models scores")
    plt.savefig(results_path + 'lights_scores.png')
    plt.close()

    print("Score of sampled lights consumptions: ", lights_model.score(X_l))
    #print("Score of real lights consumptions: " + lights_model.score(l_test[0:N]) + "\n")
