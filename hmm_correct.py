__author__ = 'Shuo Yu'
from hmmlearn import hmm
from sklearn.cross_validation import KFold
import numpy as np
import utilities
import feature_gen
import collections

N = 50
_generalized = False

def dict_inc(dict, key):
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1


# def matching_degree(hmm_model, sample, n=20):
#     """
#     Accepts an HMM model and a sample, uses a sliding window to figure out the maximum similarity (likelihood)
#     between the model and the sample.
#     :param hmm_model:
#     :param sample:
#     :param n:
#     :return: maximum likelihood
#     """
#     sample_set = [sample[x:x+n+1] for x in range(len(sample)+1-n)]
#     result_set = []
#     for s in sample_set:
#         x = hmm_model.score(np.transpose(np.matrix(s)))
#         result_set.append(x)
#     # print(result_set)
#     # print(np.amax(result_set))
#     return np.amax(result_set)


def hmm_classifier(hmm_models, sample):
    """

    :param hmm_models: a list of hmm models
    :param sample: a matrix consisting of a single sample
    :return: an integer indicating the index of the hmm model with the highest probability
    """
    # logprob = -1e9
    # index = -1
    # if not isinstance(sample, np.matrix):
    #     sample_mat = np.matrix(sample).T
    # else:
    #     sample_mat = sample
    #
    # for i in range(len(hmm_models)):
    #     model = hmm_models[i]
    #     score = model.score(sample_mat)
    #     if score > logprob:
    #         logprob = score
    #         index = i
    #
    # return index
    _pos_threshold = 1
    fall_score = -1e9
    non_score = -1e9
    fall_index = -1
    non_index = -1
    if not isinstance(sample, np.matrix):
        sample_mat = np.matrix(sample).T
    else:
        sample_mat = sample

    if _generalized: # only two models in hmm_models
        for i, model in enumerate(hmm_models):
            score = model.score(sample_mat)
            if i == 1:
                fall_score = score
                fall_index = i
            else:
                non_score = score
                non_index = i
    else: # hopefully 12 models are in hmm_models, and self._pos_model_indices is set to [8, 9, 10, 11]
        for i, model in enumerate(hmm_models):
            score = model.score(sample_mat)
            if i in [8, 9, 10, 11]:
                if score > fall_score:
                    fall_score = score
                    fall_index = i
            else:
                if score > non_score:
                    non_score = score
                    non_index = i

    # print(fall_score, non_score)
    # print(fall_index, non_index)

    # both fall_score and non_score < 0; thus, if the ratio < 1, fall_score > non_score
    if fall_score / non_score < _pos_threshold:
        return fall_index
    else:
        return non_index


def mat_to_g(value_list):
    """

    :param value_list: n x 3 matrix or list
    :return: n x 1 matrix
    """
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    mat = np.linalg.norm(mat, ord=2, axis=1, keepdims=True)
    return mat


def mat_to_vc(value_list):
    """

    :param value_list: n x 3 matrix or list
    :return: n x 2 matrix
    """
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    mat = feature_gen.calc_vt_comp_with_rem_mag(mat)
    return mat


def mat_to_peak(value_list):
    """

    :param value_list: n x 3 matrix or list
    :return: n x 1 matrix
    """
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    return feature_gen.identify_peak(mat)[0]


def sample_preprocessor(samples, func=None):
    """

    :param samples: a list of k lists, each list corresponding to the samples for a hmm model,
      each sample is a time series of data points, each data point has one or more vector components
    :param func: the function for converting the vectors, accepting n x 3 matrix or list
    :return:
    """
    if func is None:
        return samples
    return_list = []
    for i in range(len(samples)):
        return_list.append([])
        sample_type_list = samples[i] # the list of samples from the same type
        for j in range(len(sample_type_list)):
            return_list[i].append(func(sample_type_list[j]))
    return return_list


def gen_n_fold_indices(samples, n_fold):
    """
    A wrapper function for sklearn.cross_validation.KFold, performing stratified n-fold split
    :param samples:
    :param n_fold:
    :return:
    """
    indices = []
    for i in range(len(samples)):
        # for each label (or type, model)
        indices.append([])
        for training_indices, test_indices in KFold(len(samples[i]), n_fold):
            # generate n-fold indices
            indices[i].append([training_indices, test_indices])
    return indices



def threshold_model_evaluator(samples, test_samples=None, pos_model_indices=None, n_fold=0, verbose=False):
    """

    :param samples: (preprocessed) a list of k lists, each list corresponding to the results of an experiment item,
      each sample is a single-item list of peak
    :param test_samples:
    :param pos_model_indices:
    :param n_fold:
    :param generalized:
    :param verbose:
    :return:
    """

    # The threshold-based approach is always generalized.

    if n_fold == 0:
        if test_samples is None:
            test_set = samples
        else:
            test_set = test_samples

        gen_concat_samples = [[], []]
        for sample_i in range(len(samples)):
            is_positive = 1 if sample_i in pos_model_indices else 0
            concat_samples = samples[sample_i]
            if gen_concat_samples[is_positive] == []:
                gen_concat_samples[is_positive] = concat_samples
            else:
                gen_concat_samples[is_positive] = np.concatenate((gen_concat_samples[is_positive], concat_samples))
        print(gen_concat_samples)
        exit(0)

    else:
        n_fold_indices = gen_n_fold_indices(samples, n_fold)

        # do the n-fold cross-validaion
        spec = []
        sens = []
        for k in range(n_fold):
            # separate the samples into training and test
            training = []
            test = []

            for i in range(len(samples)):
                current_samples = np.array(samples[i])
                training.append(current_samples[n_fold_indices[i][k][0]])
                test.append(current_samples[n_fold_indices[i][k][1]])
            s1, s2 = hmm_model_evaluator(training, test, pos_model_indices, 0, verbose)
            spec.append(s1)
            sens.append(s2)
        # print(spec)
        # print(sens)
        return np.mean(spec), np.mean(sens)


def confusion_matrix(true_list, test_list, border=8):
    confusion_dict = {}
    flag = False
    if 2 in test_list:
        flag = True
    for true_value, test_value in zip(true_list, test_list):
        if true_value not in confusion_dict:
            confusion_dict[true_value] = [0, 0]
        if flag: # not generalized
            if (test_value < border and true_value < border) or (test_value >= border and true_value >= border):
                confusion_dict[true_value][0] += 1
            else:
                confusion_dict[true_value][1] += 1
        else: # generalized
            if (test_value == 0 and true_value < border) or (test_value == 1 and true_value >= border):
                confusion_dict[true_value][0] += 1
            else:
                confusion_dict[true_value][1] += 1

    return confusion_dict



def hmm_model_evaluator(samples, test_samples=None, pos_model_indices=None, n_fold=0, n_states=3, generalized=False,
                        verbose=False, spec_sens=True, farseeing=False):
    """

    :param samples: (preprocessed) a list of k lists, each list corresponding to the samples for a hmm model,
      each sample is a time series of data points, each data point has one or more vector components
    :param pos_model_indices: a list of indices indicating positive hmm models
    :param n_fold: n-fold cross validation. If 0, the training set itself is used as test set
    :return:
    """
    hmm_models = []
    if generalized == True and not isinstance(pos_model_indices, collections.Iterable):
        raise Exception('Option generalized requires pos_model_indices')

    if n_fold == 0:
        if test_samples is None:
            test_set = samples
        else:
            test_set = test_samples
        # train hmm models
        if generalized == False:
            for samples_for_a_hmm in samples:
                lengths = [len(time_series) for time_series in samples_for_a_hmm]
                concaternated_samples = np.concatenate(samples_for_a_hmm)
                model = hmm.GaussianHMM(n_components=n_states)
                model.fit(concaternated_samples, lengths)
                hmm_models.append(model)
        else: # generalized == True
            gen_lengths = [[], []]
            gen_concat_samples = [[], []]
            for sample_i in range(len(samples)):
                is_positive = 1 if sample_i in pos_model_indices else 0
                gen_lengths[is_positive].extend([len(time_series) for time_series in samples[sample_i]])
                concat_samples = np.concatenate(samples[sample_i])
                if gen_concat_samples[is_positive] == []:
                    gen_concat_samples[is_positive] = concat_samples
                else:
                    gen_concat_samples[is_positive] = np.concatenate((gen_concat_samples[is_positive], concat_samples))
            for i in range(2):
                model = hmm.GaussianHMM(n_components=n_states)
                model.fit(gen_concat_samples[i], gen_lengths[i])
                hmm_models.append(model)
        # test hmm models
        # 1. prepare "true" tags

        # file = open('c:/hmm_models.pkl', 'wb')
        # pickle.dump(hmm_models, file, protocol=3)
        # file.close()
        hmm_index = -1
        true_tags = []
        test_tags = []
        type_dict = {0: 0, 1: 0}
        hit_dict = {0: 0, 1: 0}
        confusion_true_tags = []
        confusion_test_tags = []
        # print("farseeing")
        for samples_for_a_hmm in test_set:
            hmm_index += 1
            for time_series in samples_for_a_hmm:
                eval_index = hmm_classifier(hmm_models, np.matrix(time_series))
                confusion_true_tags.append(hmm_index)
                confusion_test_tags.append(eval_index)
                # print(hmm_index, eval_index)
                if pos_model_indices is None:
                    true_tags.append(hmm_index)
                    test_tags.append(eval_index)
                elif generalized == False:
                    if farseeing == True:
                        true_tags.append(hmm_index)
                    elif hmm_index in pos_model_indices:
                        true_tags.append(1)
                    else:
                        true_tags.append(0)
                    if eval_index in pos_model_indices:
                        test_tags.append(1)
                    else:
                        test_tags.append(0)
                elif generalized == True:
                    if farseeing == True:
                        true_tags.append(hmm_index)
                    elif hmm_index in pos_model_indices:
                        true_tags.append(1)
                    else:
                        true_tags.append(0)
                    test_tags.append(eval_index)
        print(true_tags)
        print(test_tags)
        print(confusion_matrix(confusion_true_tags, confusion_test_tags))
        # print("farseeing1")
        for i in range(len(true_tags)):
            dict_inc(type_dict, true_tags[i])
            if true_tags[i] == test_tags[i]:
                dict_inc(hit_dict, true_tags[i])


        return_list = []
        if spec_sens:
            for key in type_dict:
                if verbose:
                    print('[Type {}]: {} total, {} hit, {:.4g} accuracy'.format(key, type_dict[key], hit_dict[key], hit_dict[key] / type_dict[key]))
                    print(hit_dict)
                    print(type_dict)
                return_list.append(hit_dict[key])
        else: # use precision and recall
            # print(type_dict)
            # print(hit_dict)
            return_list = [hit_dict[1] / (hit_dict[1] + type_dict[0] - hit_dict[0]), hit_dict[1] / type_dict[1]]

        # print("farseeing2")
        return return_list
    else: # k-fold cross-validation
        n_fold_indices = gen_n_fold_indices(samples, n_fold)

        # do the n-fold cross-validaion
        spec = []
        sens = []
        for k in range(n_fold):
            # separate the samples into training and test
            training = []
            test = []

            for i in range(len(samples)):
                current_samples = np.array(samples[i])
                training.append(current_samples[n_fold_indices[i][k][0]])
                test.append(current_samples[n_fold_indices[i][k][1]])
            s1, s2 = hmm_model_evaluator(training, test, pos_model_indices, 0, n_states, generalized, verbose, spec_sens)
            spec.append(s1)
            sens.append(s2)
        # print(spec)
        # print(sens)
        return (spec, sens)


def discretize(list):
    ret_list = []
    for element in list:
        if element < 500:
            ret_list.append(0)
        elif element < 700:
            ret_list.append(1)
        elif element < 1200:
            ret_list.append(2)
        elif element < 2500:
            ret_list.append(3)
        elif element < 4000:
            ret_list.append(4)
        else: # >= 4000
            ret_list.append(5)
    return ret_list


def load_samples_from_db(cur=None, sensor_id=None):
    """

    :param sensor_id:
    :return: a list of non_lists, fall_lists
    """


    non_lists = []
    fall_lists = []

    start, end = 1, 6
    if sensor_id is not None:
        start = sensor_id
        end = sensor_id + 1

    arg_dict = {
        "sensor_id": 1, # 1 to 5
        "subject_id": 1, # 1 to 5
        "label_id": 1, # 1 to 4
        "freq": 12.5,
        "db_name": "test_data_fall_1_re",
    }

    # Four-direction falls (j), five trials each (i)
    # corrected after refactor: Four-direction falls (i), five trials each (j)
    for i in range(1, 5):
        fall_lists.append([])
        for j in range(1, 6):
            for k in range(start, end):
                extended = True
                arg_dict["db_name"] = "test_data_fall_1_re"
                arg_dict["sensor_id"] = k
                arg_dict["label_id"] = i
                arg_dict["subject_id"] = j
                sample = utilities.read_data_from_db(cur, **arg_dict)
                sample = feature_gen.sample_around_peak(np.matrix(sample), 25, 25).tolist()
                sample2 = []
                if extended:
                    arg_dict["db_name"] = "test_data_fall_1b"
                    sample2 = utilities.read_data_from_db(cur, **arg_dict)
                    sample2 = feature_gen.sample_around_peak(np.matrix(sample2), 25, 25).tolist()

                # sample is a 2-D list

                # sample = discretize(sample)

                # if arg_dict["label_id"] <= 3:
                #     sample = sample[:-60]
                # adding_sample = sample[start_indices[index] - 2 :]
                adding_sample = sample + sample2
                if len(adding_sample) != 0:
                    fall_lists[i-1].append(adding_sample)
            # index += 1

    # print(len(sample_lists))

    arg_dict2 = {
        "sensor_id": 1, # 1 to 5
        "subject_id": 1, # 1 to 5
        "label_id": 1, # 1 to 8
        "freq": 12.5,
        "db_name": "test_data_stage_1",
    }

    # Eight ADLs (i), five subjects each (j)
    for i in range(1, 9):
        non_lists.append([])
        for j in range(1, 6):
            for k in range(start, end):
                extended = False
                arg_dict2["db_name"] = "test_data_stage_1"
                arg_dict2["sensor_id"] = k
                arg_dict2["label_id"] = i
                arg_dict2["subject_id"] = j
                sample = utilities.read_data_from_db(cur, **arg_dict2)
                sample2 = []
                if extended:
                    arg_dict2["db_name"] = "test_data_stage_1b"
                    sample2 = utilities.read_data_from_db(cur, **arg_dict2)
                # sample = utilities.mat_to_g(sample).tolist()
                # sample = discretize(sample)
                adding_sample = sample + sample2
                if len(adding_sample) != 0:
                    non_lists[i-1].append(adding_sample)

    return non_lists, fall_lists


def load_fall_samples_from_farseeing(cur, interval=5000):
    sql = 'SELECT subject_id, timestamp FROM test_data_farseeing where label_id != 0;'
    cur.execute(sql)
    sub_ts_dict = {}
    for row in cur:
        subject_id = int(row[0])
        timestamp = int(row[1])
        sub_ts_dict[subject_id] = timestamp

    ret_list = []
    counter = 0
    for sub in sub_ts_dict:
        ret_list.append([])
        sql = '''SELECT timestamp, x_accel, y_accel, z_accel
            FROM test_data_farseeing
            WHERE subject_id = '{0}' AND timestamp >= {1} - {2} AND timestamp < {1} + {2}
        '''.format(sub, sub_ts_dict[sub], interval)
        cur.execute(sql)
        for row in cur:
            ret_list[counter].append([int(row[1]), int(row[2]), int(row[3])])
            # ret_list[counter].append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
        counter += 1

    return ret_list


def load_non_samples_from_farseeing(cur, interval=5000, before_after=60000):
    sql = 'SELECT subject_id, timestamp FROM test_data_farseeing where label_id != 0;'
    cur.execute(sql)
    sub_ts_dict = {}
    for row in cur:
        subject_id = int(row[0])
        timestamp = int(row[1])
        sub_ts_dict[subject_id] = timestamp

    ret_list = []
    counter = 0
    for sub in sub_ts_dict:
        start_ts = sub_ts_dict[sub] - before_after - interval
        cur_ts = start_ts
        end_ts = sub_ts_dict[sub] + before_after - interval

        while cur_ts < end_ts:
            if cur_ts < sub_ts_dict[sub] and cur_ts + interval * 2 > sub_ts_dict[sub]: # skip segment with fall
                cur_ts += interval * 2
                continue

            ret_list.append([])
            sql = '''SELECT timestamp, x_accel, y_accel, z_accel
                FROM test_data_farseeing
                WHERE subject_id = '{0}' AND timestamp >= {1} AND timestamp < {2}
            '''.format(sub, cur_ts, cur_ts + interval * 2)
            cur.execute(sql)
            for row in cur:
                # ret_list[counter].append([int(row[0]), int(row[1]), int(row[2]), int(row[3])])
                ret_list[counter].append([int(row[1]), int(row[2]), int(row[3])])
            counter += 1
            # if counter % 100 == 0:
            #     print(counter)
            cur_ts += interval * 2

    return ret_list


def decompose_threshold(samples, threshold):
    ret_dict = {}
    for i, category in enumerate(samples):
        ret_dict[i] = [0, 0]
        for sample in category:
            peak_mag = np.linalg.norm(mat_to_peak(np.matrix(sample)))
            if i < 8 and peak_mag < 2500 or i >=8 and peak_mag >= 2500:
                ret_dict[i][0] += 1
            else:
                ret_dict[i][1] += 1
    return ret_dict

if __name__ == '__main__':
    cur = utilities.db_connect()

    # start_indices = [20, 18, 18, 17, 19, 18, 17, 18, 19, 17, 17, 18, 16, 15, 16, 14, 17, 14, 11, 9]

    do_pos_5 = True
    do_pos_1 = False
    spec_sens = True
    root = 'c:/_test_space/hmm_fall_detection_farseeing2/'

    # non_lists, fall_lists = load_samples_from_db()
    #
    # prep_lists = sample_preprocessor(non_lists + fall_lists, mat_to_peak)
    # print(prep_lists)
    # threshold_model_evaluator(prep_lists, None, [8, 9, 10, 11], 0, False)
    n_loops = 1
    max_n_states = 5
    pos_model_indices = [8, 9, 10, 11]
    use_metrics = [None, mat_to_g, mat_to_vc]
    is_gen = [False, True]

    if do_pos_5:
        non_lists, fall_lists = load_samples_from_db(cur)
        print([len(x) for x in non_lists + fall_lists])
        # print(decompose_threshold(non_lists+fall_lists, 2500))
        # exit(2)
        non_farseeing = load_non_samples_from_farseeing(cur, interval=500, before_after=60000)
        fall_farseeing = load_fall_samples_from_farseeing(cur, interval=500)
        print([len(x) for x in [non_farseeing] + [fall_farseeing]])
        outputs = [open(root + 'pos_5_sep.txt', 'a'), open(root + 'pos_5_gen.txt', 'a')]

        for m in range(3): # three metrics
            prep_lists = sample_preprocessor(non_lists + fall_lists, use_metrics[m])
            prep_farseeing = sample_preprocessor([non_farseeing] + [fall_farseeing], use_metrics[m])
            print('Sample preprocessed')
            for y in range(2): # not generalized & generalized
                if y == 0:
                    _generalized = False
                else:
                    _generalized = True
                for n in range(4, max_n_states): # number of hidden states for HMM
                    accs = [0] * 2
                    stds = [0] * 2
                    for p in range(n_loops):
                        results = hmm_model_evaluator(prep_lists, prep_farseeing, pos_model_indices=pos_model_indices, n_states=n,
                                                      generalized=is_gen[y], n_fold=0, spec_sens=spec_sens, farseeing=True)
                        # results = hmm_model_evaluator(prep_lists, pos_model_indices=pos_model_indices, n_states=n,
                        #       generalized=is_gen[y], n_fold=0, spec_sens=spec_sens, farseeing=False)
                        outputs[y].write('  >> {}, {}\n'.format(*results))
                        for q in range(len(results)):
                            accs[q] += np.mean(results[q])
                            stds[q] += np.std(results[q])
                    for q in range(len(results)):
                        accs[q] /= n_loops
                        stds[q] /= n_loops
                    output_str = 'n = {}, metric = {}, Specificity: {:.3g} ({:.3g}), Sensitivity: {:.3g} ({:.3g})\n'.format(
                        n, m, accs[0] / 2618, stds[0], accs[1] / 22, stds[1])
                    print(output_str.rstrip())
                    outputs[y].write(output_str)
                    outputs[y].flush()
    ################################################################################

    if do_pos_1:
        for k in range(1, 6):
            outputs = [open(root + 'pos_1_sep.txt', 'a'), open(root + 'pos_1_gen.txt', 'a')]
            outputs[0].write('##### Sensor ID = {} #####\n'.format(k))
            outputs[1].write('##### Sensor ID = {} #####\n'.format(k))
            non_lists, fall_lists = load_samples_from_db(cur, k)
            for m in range(3): # three metrics
                prep_lists = sample_preprocessor(non_lists + fall_lists, use_metrics[m])

                for y in range(2): # not generalized & generalized

                    for n in range(3, max_n_states): # number of hidden states for HMM
                        accs = [0] * 2
                        stds = [0] * 2
                        for p in range(n_loops):
                            try:
                                results = hmm_model_evaluator(prep_lists,
                                                              pos_model_indices=pos_model_indices,
                                                              n_states=n,
                                                              generalized=is_gen[y], n_fold=5, spec_sens=spec_sens)
                                outputs[y].write('  >> {}, {}\n'.format(*results))
                            except Exception:
                                results = hmm_model_evaluator(prep_lists,
                                                              pos_model_indices=pos_model_indices,
                                                              n_states=n,
                                                              generalized=is_gen[y], n_fold=3, spec_sens=spec_sens)
                                outputs[y].write('  >> {}, {}\n'.format(*results))
                            for q in range(len(results)):
                                accs[q] += np.mean(results[q])
                                stds[q] += np.std(results[q])
                        for q in range(len(results)):
                            accs[q] /= n_loops
                            stds[q] /= n_loops
                        output_str = 'n = {}, metric = {}, [0]: {:.4g} ({:.4g}), [1]: {:.4g} ({:.4g})\n'.format(
                            n, m, accs[0], stds[0], accs[1], stds[1])
                        print(output_str.rstrip())
                        outputs[y].write(output_str)
                        outputs[y].flush()

                # print ([len(everything) for everything in prep_lists])
                # for a in range(len(prep_lists)):
                #     for b in range(len(prep_lists[a])):
                #         print('a = {}, b = {}, first fifteen: {}'.format(a, b, prep_lists[a][b][:15]))