__author__ = 'Shuo Yu'

import pickle
import hmm_correct
import utilities
import fall_detect_portal_build

hmm_model_path = 'c:/hmm_models.pkl'
hmm_file = open(hmm_model_path, 'rb')
hmm_models = pickle.load(hmm_file)
hmm_file.close()


def classify(instance, hmm_models, n=8):
    """

    :param instance: a n x 3 matrix denoting a snippet of 3-axial acceleration data
    :param hmm_models: the pickled hmms
    :param n: this param denotes the number of non-fall hmms. by default, 12 hmms are trained for different categories
              of falls and non-falls. The first 8 are non-falls. Adjust this param if the number of hmms has changed.
    :return: True or False, denoting whether or not a fall exists.
    """
    converted_instance = hmm_correct.mat_to_vc(instance)
    index = hmm_correct.hmm_classifier(hmm_models, converted_instance)
    return False if index < n else True


if __name__ == '__main__':
    cur = utilities.db_connect()
    n_states = 6
    non_lists, fall_lists = hmm_correct.load_samples_from_db(cur)
    prep_lists = hmm_correct.sample_preprocessor(non_lists + fall_lists, hmm_correct.mat_to_vc)
    # print(hmm_correct.hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=6, generalized=False, n_fold=0, spec_sens=True))
    # exit(0)

    # hmm_models = fall_detect_portal_build.hmm_build(prep_lists, n_states)

    n_category = -1
    total = 0
    hit = 0
    for category in prep_lists:
        n_category += 1
        for instance in category:
            total += 1
            result = classify(instance, hmm_models)
            if (not result and (n_category < 8)) or (result and (n_category >= 8)):
                hit += 1
    print('{} / {}'.format(hit, total))