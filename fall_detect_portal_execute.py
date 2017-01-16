__author__ = 'Shuo Yu'

import pickle
import hmm_correct
import utilities
import numpy as np
import fall_detect_portal_build

hmm_model_path = 'c:/hmm_models.pkl'
try:
    with open(hmm_model_path, 'rb') as hmm_file:
        hmm_models = pickle.load(hmm_file)
except Exception as e:
    print('Pickled HMMs read error: ', e)
    exit(-1)


def classify(instance, hmm_models, n=8):
    """

    :param instance: a n x 3 matrix denoting a snippet of 3-axial acceleration data
    :param hmm_models: the pickled hmms
    :param n: this param denotes the number of non-fall hmms. by default, 12 hmms are trained for different categories
              of falls and non-falls. The first 8 are non-falls. Adjust this param if the number of hmms has changed.
    :return: 1 or 0, denoting whether (1) or not (0) a fall exists.
    """
    instance_0 = instance[0]
    converted_instance = []
    if len(instance_0) == 3: # without timestamp
        converted_instance = hmm_correct.mat_to_vc(instance)
    elif len(instance_0) == 4: # with timestamp
        converted_instance = hmm_correct.mat_to_vc(np.matrix(instance)[:, 1:4])
    index = hmm_correct.hmm_classifier(hmm_models, converted_instance)
    return 0 if index < n else 1


def classify_with_ts(instance, hmm_models, interval=10000, freq=None, fall_seg_thres=1, n=8):
    """
    Segment the instance into segments. Detect falls from those segments.
    :param instance: a n x 4 matrix denoting a snippet of timestamp and 3-axial acceleration data
    :param hmm_models: the pickled hmms. change 'hmm_model_path' to locate the proper file for pickled hmms.
    :param interval: define the interval to segment the input data, in milliseconds. E.g., for a snippet of 120 secs,
              setting interval to 10000 will segment the data into 12 segments, each independently detected for falls.
    :param fall_seg_thres: the number of fall segments that is needed to declare a fall detected. 1 by default.
    :param freq: the frequency of sample collected. Set it to None to let the script infer the frequency from the first
              two data points of the instance. Set it to a fixed value if the data stream is unstable.
    :param n: this param denotes the number of non-fall hmms. by default, 12 hmms are trained for different categories
              of falls and non-falls. The first 8 are non-falls by default.
              Adjust this param if the number of hmms has changed.
    :return: 1 or 0, denoting whether (1) or not (0) a fall exists.
    """
    try:
        peak_mag = np.linalg.norm(hmm_correct.mat_to_peak(np.matrix(instance)[:, 1:4]))
        if peak_mag < 2200:
            return 0

        # freq_inv is the interval between two consecutive timestamps, in milliseconds
        if freq is None:
            freq_inv = instance[1][0] - instance[0][0]
        else:
            freq_inv = int(1000 / freq)
        cur_index = 0
        end_index = len(instance)

        index_skip = interval // freq_inv

        num_fall_seg = 0
        while cur_index < end_index:
            # if cur_index + index_skip * 2 > end_index:
            #     is_fall = classify(instance[cur_index:], hmm_models)
            # else:
            is_fall = classify(instance[cur_index:cur_index+index_skip], hmm_models, n=n)
            num_fall_seg += is_fall
            cur_index += index_skip

        return 1 if num_fall_seg >= fall_seg_thres else 0
    except Exception as e:
        return 0


if __name__ == '__main__':
    cur = utilities.db_connect()

    fall_seg_thres = 1

    fall_samples = hmm_correct.load_fall_samples_from_farseeing(cur, interval=500)
    non_samples = hmm_correct.load_non_samples_from_farseeing(cur, interval=500, before_after=60000)

    tp = 0
    tn = 0
    print('{0}, {1}'.format(len(fall_samples), len(non_samples)))
    for sample in fall_samples:
        if classify_with_ts(sample, hmm_models, interval=1000, fall_seg_thres=fall_seg_thres) == 1:
            tp += 1
    print('TP rate: {}'.format(tp / len(fall_samples)))
    for sample in non_samples:
        if classify_with_ts(sample, hmm_models, interval=1000, fall_seg_thres=fall_seg_thres) == 0:
            tn += 1

    print('TP rate: {}, TN rate: {}'.format(tp / len(fall_samples), tn / len(non_samples)))
    # n_states = 6
    # non_lists, fall_lists = hmm_correct.load_samples_from_db(cur)
    # prep_lists = hmm_correct.sample_preprocessor(non_lists + fall_lists, hmm_correct.mat_to_vc)
    # # print(hmm_correct.hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=6, generalized=False, n_fold=0, spec_sens=True))
    # # exit(0)
    #
    # # hmm_models = fall_detect_portal_build.hmm_build(prep_lists, n_states)
    #
    # n_category = -1
    # total = 0
    # hit = 0
    # for category in prep_lists:
    #     n_category += 1
    #     for instance in category:
    #         total += 1
    #         result = classify(instance, hmm_models)
    #         if (not result and (n_category < 8)) or (result and (n_category >= 8)):
    #             hit += 1
    # print('{} / {}'.format(hit, total))