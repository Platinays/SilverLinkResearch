__author__ = 'Shuo Yu'
from sklearn.externals import joblib
from hmmlearn import hmm
from sklearn.cross_validation import KFold
import numpy as np
import utilities
import feature_gen
import collections

N = 50

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
    logprob = -1e9
    index = -1
    if not isinstance(sample, np.matrix):
        sample_mat = np.matrix(sample).T
    else:
        sample_mat = sample

    for i in range(len(hmm_models)):
        model = hmm_models[i]
        score = model.score(sample_mat)
        if score > logprob:
            logprob = score
            index = i

    return index


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


def f_score(pr, rc):
    return 2 * pr * rc / (pr + rc) if pr + rc > 0 else 0


def thres_calc_pr_rc(samples, thres):
    true_pos = 0
    true_neg = 0
    for value in samples[0]:
        if value < thres:
            true_neg += 1
    for value in samples[1]:
        if value > thres:
            true_pos += 1
    print('{}, (-): {} / {}, (+): {} / {}'.format(thres, true_neg, len(samples[0]), true_pos, len(samples[1])))
    pr = true_pos / (true_pos + len(samples[0]) - true_neg) if (true_pos + len(samples[0]) - true_neg) != 0 else 0
    rc = true_pos / len(samples[1])
    # exit(0)
    return pr, rc


def generalize(samples, pos_model_indices):
    gen_concat_samples = [[], []]
    for sample_i in range(len(samples)):
        is_positive = 1 if sample_i in pos_model_indices else 0
        concat_samples = samples[sample_i]
        if gen_concat_samples[is_positive] == []:
            gen_concat_samples[is_positive] = concat_samples
        else:
            gen_concat_samples[is_positive] = np.concatenate((gen_concat_samples[is_positive], concat_samples))
    return gen_concat_samples


def search_best_thres(samples):
    """

    :param samples: a list of two lists with the first negative (-) samples and the second positive (+) samples
    :param metric:
    :return:
    """
    best_thres = 0
    best_met = 0
    for thres in range(1000, 4100, 100):
        pr, rc = thres_calc_pr_rc(samples, thres)
        met = f_score(pr, rc)
        if met > best_met:
            best_thres = thres
            best_met = met
    return best_thres


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

        # search for a best threshold to discriminate the two sets
        best_thres = search_best_thres(generalize(samples, pos_model_indices))
        pr, rc = thres_calc_pr_rc(generalize(test_set, pos_model_indices), best_thres)
        return pr, rc, best_thres

    else:
        n_fold_indices = gen_n_fold_indices(samples, n_fold)

        # do the n-fold cross-validaion
        spec = []
        sens = []
        thres_list = []
        for k in range(n_fold):
            # separate the samples into training and test
            training = []
            test = []

            for i in range(len(samples)):
                current_samples = np.array(samples[i])
                training.append(current_samples[n_fold_indices[i][k][0]])
                test.append(current_samples[n_fold_indices[i][k][1]])


            s1, s2, s3 = threshold_model_evaluator(training, test, pos_model_indices, 0, verbose)
            spec.append(s1)
            sens.append(s2)
            thres_list.append(s3)
        # print(spec)
        # print(sens)
        return spec, sens, thres_list


def calc_euclid_dist(mat_1, mat_2):
    return np.mean(np.linalg.norm(mat_1 - mat_2, ord=2, axis=1))


def knn_classifier(lazy_models, sample, k=1):
    n_samples_model = [len(e) for e in lazy_models]
    flat_list = np.concatenate(lazy_models)
    min_dist_list = [1e6] * k
    indices = [-1] * k
    # print(k)
    for i in range(len(flat_list)):
        dist = calc_euclid_dist(flat_list[i], sample)
        for j in range(k):
            # print(dist, min_dist_list[j])
            if dist < min_dist_list[j]:
                # print(min_dist_list)
                min_dist_list.insert(j, dist)
                # print(min_dist_list)
                min_dist_list.pop()
                # print(min_dist_list)

                # print(indices)
                indices.insert(j, 0 if i < n_samples_model[0] else 1)
                # indices.insert(j, i)
                # print(indices)
                indices.pop()
                # print(indices)
                break

    return 0 if np.mean(indices) < 0.5 else 1


def knn_model_evaluator(samples, test_samples=None, pos_model_indices=None, n_fold=0, k_nearest=1, verbose=False):
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

        # search for a best threshold to discriminate the two sets
        gen_samples = generalize(samples, pos_model_indices)
        gen_tests = generalize(test_set, pos_model_indices)

        trues = [0, 0]
        for i in range(2):
            for item in gen_tests[i]:
                if knn_classifier(gen_samples, item, k_nearest) == i:
                    trues[i] += 1

        pr = trues[1] / (trues[1] + len(gen_tests[0]) - trues[0]) if trues[1] + len(gen_tests[0]) - trues[0] != 0 else 0
        rc = trues[1] / len(gen_tests[1])
        print('{} / {}, {} / {}'.format(trues[0], len(gen_tests[0]), trues[1], len(gen_tests[1])))
        return pr, rc

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

            s1, s2= knn_model_evaluator(training, test, pos_model_indices, n_fold=0, k_nearest=k_nearest, verbose=verbose)
            spec.append(s1)
            sens.append(s2)
        # print(spec)
        # print(sens)
        return spec, sens


def hmm_model_evaluator(samples, test_samples=None, pos_model_indices=None, n_fold=0, n_states=3, generalized=False, verbose=False, spec_sens=True):
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
        hmm_index = -1
        true_tags = []
        test_tags = []
        type_dict = {0: 0, 1: 0}
        hit_dict = {0: 0, 1: 0}
        for samples_for_a_hmm in test_set:
            hmm_index += 1
            for time_series in samples_for_a_hmm:
                eval_index = hmm_classifier(hmm_models, np.matrix(time_series))
                if pos_model_indices is None:
                    true_tags.append(hmm_index)
                    test_tags.append(eval_index)
                elif generalized == False:
                    if hmm_index in pos_model_indices:
                        true_tags.append(1)
                    else:
                        true_tags.append(0)
                    if eval_index in pos_model_indices:
                        test_tags.append(1)
                    else:
                        test_tags.append(0)
                elif generalized == True:
                    if hmm_index in pos_model_indices:
                        true_tags.append(1)
                    else:
                        true_tags.append(0)
                    test_tags.append(eval_index)

        # print(true_tags)
        # print(test_tags)

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
                return_list.append(hit_dict[key] / type_dict[key])
        else: # use precision and recall
            # print('{} / {}, {} / {}'.format(hit_dict[1], hit_dict[1] + type_dict[0] - hit_dict[0], hit_dict[1], type_dict[1]))
            return_list = [hit_dict[1] / (hit_dict[1] + type_dict[0] - hit_dict[0])
                           if hit_dict[1] + type_dict[0] - hit_dict[0] != 0 else 0,
                           hit_dict[1] / type_dict[1]]

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


def load_samples_from_db(sensor_id=None):
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
                arg_dict["sensor_id"] = k
                arg_dict["label_id"] = i
                arg_dict["subject_id"] = j
                sample = utilities.read_data_from_db(cur, **arg_dict)
                sample = feature_gen.sample_around_peak(np.matrix(sample), 25, 25)

                # sample is a 2-D list

                # sample = discretize(sample)

                # if arg_dict["label_id"] <= 3:
                #     sample = sample[:-60]
                # adding_sample = sample[start_indices[index] - 2 :]
                adding_sample = sample
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
                arg_dict2["sensor_id"] = k
                arg_dict2["label_id"] = i
                arg_dict2["subject_id"] = j
                sample = utilities.read_data_from_db(cur, **arg_dict2)
                if len(sample) == 0:
                    continue
                sample = feature_gen.sample_around_peak(np.matrix(sample), 25, 25)
                non_lists[i-1].append(sample)
                # sample = utilities.mat_to_g(sample).tolist()
                # sample = discretize(sample)

    return non_lists, fall_lists


def execute_hmm():
    cur = utilities.db_connect()

    # start_indices = [20, 18, 18, 17, 19, 18, 17, 18, 19, 17, 17, 18, 16, 15, 16, 14, 17, 14, 11, 9]

    do_pos_5 = True
    do_pos_1 = True
    spec_sens = False
    root = 'c:/_test_space/hmm_fall_detection_corrected/pr_rc/'

    # non_lists, fall_lists = load_samples_from_db()
    #
    # prep_lists = sample_preprocessor(non_lists + fall_lists, mat_to_peak)
    # print(prep_lists)
    # threshold_model_evaluator(prep_lists, None, [8, 9, 10, 11], 0, False)
    n_loops = 1
    max_n_states = 7
    pos_model_indices = [8, 9, 10, 11]
    use_metrics = [None, mat_to_g, mat_to_vc]
    is_gen = [False, True]

    if do_pos_5:
        non_lists, fall_lists = load_samples_from_db()
        outputs = [open(root + 'pos_5_sep.txt', 'a'), open(root + 'pos_5_gen.txt', 'a')]

        for m in range(3): # three metrics
            prep_lists = sample_preprocessor(non_lists + fall_lists, use_metrics[m])
            print('Sample preprocessed')
            for y in range(2): # not generalized & generalized
                for n in range(3, max_n_states): # number of hidden states for HMM
                    accs = [0] * 2
                    stds = [0] * 2
                    for p in range(n_loops):
                        results = hmm_model_evaluator(prep_lists, pos_model_indices=pos_model_indices, n_states=n,
                                                      generalized=is_gen[y], n_fold=10, spec_sens=spec_sens)
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
    ################################################################################

    if do_pos_1:
        for k in range(1, 6):
            outputs = [open(root + 'pos_1_sep.txt', 'a'), open(root + 'pos_1_gen.txt', 'a')]
            outputs[0].write('##### Sensor ID = {} #####\n'.format(k))
            outputs[1].write('##### Sensor ID = {} #####\n'.format(k))
            non_lists, fall_lists = load_samples_from_db(k)
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


def execute_thres():
    cur = utilities.db_connect()
    do_pos_5 = True
    do_pos_1 = True
    n_loops = 1
    pos_model_indices = [8, 9, 10, 11]
    y = 0
    root = 'c:/_test_space/thres_fall_detection/'
    if do_pos_5:
        non_lists, fall_lists = load_samples_from_db()
        outputs = [open(root + 'pos_5.txt', 'a')]

        prep_lists = sample_preprocessor(non_lists + fall_lists, mat_to_peak)
        print('Sample preprocessed')
        accs = [0] * 2
        stds = [0] * 2
        thress = []
        for p in range(n_loops):
            results = threshold_model_evaluator(prep_lists, pos_model_indices=pos_model_indices, n_fold=10)

            outputs[y].write('  >> {}, {}, {}\n'.format(*results))
            for q in range(2):
                accs[q] += np.mean(results[q])
                stds[q] += np.std(results[q])
            thress.append(results[2])


        for q in range(2):
            accs[q] /= n_loops
            stds[q] /= n_loops
        output_str = '[0]: {:.4g} ({:.4g}), [1]: {:.4g} ({:.4g}), {}\n'.format(
            accs[0], stds[0], accs[1], stds[1], thress)
        print(output_str.rstrip())
        outputs[y].write(output_str)
        outputs[y].flush()
    if do_pos_1:
        for k in range(1, 6):
            non_lists, fall_lists = load_samples_from_db(k)
            outputs = [open(root + 'pos_1.txt', 'a')]
            outputs[0].write('##### Sensor ID = {} #####\n'.format(k))

            prep_lists = sample_preprocessor(non_lists + fall_lists, mat_to_peak)
            print('Sample preprocessed')
            accs = [0] * 2
            stds = [0] * 2
            thress = []
            for p in range(n_loops):
                try:
                    results = threshold_model_evaluator(prep_lists, pos_model_indices=pos_model_indices, n_fold=5)
                except Exception:
                    results = threshold_model_evaluator(prep_lists, pos_model_indices=pos_model_indices, n_fold=3)
                outputs[y].write('  >> {}, {}, {}\n'.format(*results))
                for q in range(2):
                    accs[q] += np.mean(results[q])
                    stds[q] += np.std(results[q])
                thress.append(results[2])
            for q in range(2):
                accs[q] /= n_loops
                stds[q] /= n_loops
            output_str = '[0]: {:.4g} ({:.4g}), [1]: {:.4g} ({:.4g}), {}\n'.format(
                accs[0], stds[0], accs[1], stds[1], thress)
            print(output_str.rstrip())
            outputs[y].write(output_str)
            outputs[y].flush()


if __name__ == '__main__':
    cur = utilities.db_connect()

    # start_indices = [20, 18, 18, 17, 19, 18, 17, 18, 19, 17, 17, 18, 16, 15, 16, 14, 17, 14, 11, 9]

    do_pos_5 = True
    do_pos_1 = True
    spec_sens = False
    root = 'c:/_test_space/knn_fall_detection/'

    # non_lists, fall_lists = load_samples_from_db()
    #
    # prep_lists = sample_preprocessor(non_lists + fall_lists, mat_to_peak)
    # print(prep_lists)
    # threshold_model_evaluator(prep_lists, None, [8, 9, 10, 11], 0, False)
    n_loops = 1
    pos_model_indices = [8, 9, 10, 11]
    use_metrics = [None, mat_to_g, mat_to_vc]
    is_gen = [False, True]
    y = 0
    for k_nearest in range(1, 6, 2):
        if do_pos_5:
            non_lists, fall_lists = load_samples_from_db()
            outputs = [open(root + 'pos_5_k{}.txt'.format(k_nearest), 'a')]

            for m in range(3): # three metrics
                prep_lists = sample_preprocessor(non_lists + fall_lists, use_metrics[m])
                # print('Sample preprocessed')
                # print([len(x) for x in prep_lists])
                accs = [0] * 2
                stds = [0] * 2
                for p in range(n_loops):
                    results = knn_model_evaluator(prep_lists, pos_model_indices=pos_model_indices,
                                                  n_fold=10, k_nearest=k_nearest)
                    outputs[y].write('  >> {}, {}\n'.format(*results))
                    for q in range(len(results)):
                        accs[q] += np.mean(results[q])
                        stds[q] += np.std(results[q])
                for q in range(len(results)):
                    accs[q] /= n_loops
                    stds[q] /= n_loops
                output_str = 'metric = {}, [0]: {:.4g} ({:.4g}), [1]: {:.4g} ({:.4g})\n'.format(
                    m, accs[0], stds[0], accs[1], stds[1])
                print(output_str.rstrip())
                outputs[y].write(output_str)
                outputs[y].flush()
        ################################################################################

        if do_pos_1:
            for k in range(1, 6):
                outputs = [open(root + 'pos_1_k{}.txt'.format(k_nearest), 'a')]
                outputs[0].write('##### Sensor ID = {} #####\n'.format(k))
                non_lists, fall_lists = load_samples_from_db(k)
                for m in range(3): # three metrics
                    prep_lists = sample_preprocessor(non_lists + fall_lists, use_metrics[m])

                    accs = [0] * 2
                    stds = [0] * 2
                    for p in range(n_loops):
                        try:
                            results = knn_model_evaluator(prep_lists,
                                                          pos_model_indices=pos_model_indices,
                                                          n_fold=5, k_nearest=k_nearest)
                            outputs[y].write('  >> {}, {}\n'.format(*results))
                        except Exception:
                            results = knn_model_evaluator(prep_lists,
                                                          pos_model_indices=pos_model_indices,
                                                          n_fold=3, k_nearest=k_nearest)
                            outputs[y].write('  >> {}, {}\n'.format(*results))
                        for q in range(len(results)):
                            accs[q] += np.mean(results[q])
                            stds[q] += np.std(results[q])
                    for q in range(len(results)):
                        accs[q] /= n_loops
                        stds[q] /= n_loops
                    output_str = 'metric = {}, [0]: {:.4g} ({:.4g}), [1]: {:.4g} ({:.4g})\n'.format(
                        m, accs[0], stds[0], accs[1], stds[1])
                    print(output_str.rstrip())
                    outputs[y].write(output_str)
                    outputs[y].flush()