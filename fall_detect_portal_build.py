__author__ = 'Shuo Yu'

import hmm_correct
import utilities
import numpy as np
import pickle
from hmmlearn import hmm


def hmm_build(samples, n_states):
    hmm_models = []
    for samples_for_a_hmm in samples:
        lengths = [len(time_series) for time_series in samples_for_a_hmm]
        concaternated_samples = np.concatenate(samples_for_a_hmm)
        model = hmm.GaussianHMM(n_components=n_states)
        model.fit(concaternated_samples, lengths)
        hmm_models.append(model)
    return hmm_models


if __name__ == '__main__':
    cur = utilities.db_connect()
    n_states = 6
    non_lists, fall_lists = hmm_correct.load_samples_from_db(cur)
    prep_lists = hmm_correct.sample_preprocessor(non_lists + fall_lists, hmm_correct.mat_to_vc)
    hmm_models = hmm_build(prep_lists, n_states)
    file = open('c:/hmm_models.pkl', 'wb')
    pickle.dump(hmm_models, file, protocol=3)
    file.close()