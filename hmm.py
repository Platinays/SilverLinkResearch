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


def threshold_model_evaluator(samples, test_samples=None, pos_model_indices=None, n_fold=0, n_states=3, generalized=False, verbose=False):
    if n_fold == 0:
        pass
    else:
        n_fold_indices = []
        for i in range(len(samples)):
            # for each label (or type, model)
            n_fold_indices.append([])
            for training_indices, test_indices in KFold(len(samples[i]), n_fold):
                # generate n-fold indices
                n_fold_indices[i].append([training_indices, test_indices])

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
            s1, s2 = hmm_model_evaluator(training, test, pos_model_indices, 0, n_states, generalized, verbose)
            spec.append(s1)
            sens.append(s2)
        # print(spec)
        # print(sens)
        return np.mean(spec), np.mean(sens)


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
            return_list = [hit_dict[1] / (hit_dict[1] + type_dict[0] - hit_dict[0]), hit_dict[1] / type_dict[1]]

        return return_list
    else: # k-fold cross-validation
        n_fold_indices = []
        for i in range(len(samples)):
            # for each label (or type, model)
            n_fold_indices.append([])
            for training_indices, test_indices in KFold(len(samples[i]), n_fold):
                # generate n-fold indices
                n_fold_indices[i].append([training_indices, test_indices])

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
        return np.mean(spec), np.mean(sens)


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


# def discretize(list):
#     ret_list = []
#     for element in list:
#         if element < 250:
#             ret_list.append(0)
#         elif element < 500:
#             ret_list.append(1)
#         elif element < 700:
#             ret_list.append(2)
#         elif element < 1200:
#             ret_list.append(3)
#         elif element < 2000:
#             ret_list.append(4)
#         elif element < 2800:
#             ret_list.append(5)
#         elif element < 3600:
#             ret_list.append(6)
#         else:
#             ret_list.append(7)
#     return ret_list


if __name__ == '__main__':
    cur = utilities.db_connect()

    # start_indices = [20, 18, 18, 17, 19, 18, 17, 18, 19, 17, 17, 18, 16, 15, 16, 14, 17, 14, 11, 9]
    do_pos_5 = False
    do_pos_1 = True
    spec_sens = False
    root = 'c:/_test_space/hmm_fall_detection/pr_rc/'
    # for k in range(1, 6):
    if do_pos_5:
        sample_lists = []
        test_lists = []
        # print('##### Sensor ID = {} #####'.format(k))
        arg_dict = {
            "sensor_id": 1, # 1 to 5
            "subject_id": 0, # 0 to 3
            "label_id": 1, # 1 to 5
            "freq": 12.5,
            "db_name": "test_data_fall_1",
        }

        index = 0
        # Four-direction falls (j), five trials each (i)
        for j in range(0, 4):
            sample_lists.append([])
            for i in range(1, 6):
                for k in range(1, 6):
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
                        sample_lists[j].append(adding_sample)
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
            test_lists.append([])
            for j in range(1, 6):
                for k in range(1, 6):
                    arg_dict2["sensor_id"] = k
                    arg_dict2["label_id"] = i
                    arg_dict2["subject_id"] = j
                    sample = utilities.read_data_from_db(cur, **arg_dict2)
                    # sample = utilities.mat_to_g(sample).tolist()
                    # sample = discretize(sample)
                    if len(sample) != 0:
                        test_lists[i-1].append(sample)
        n_loops = 5
        use_metrics = [None, mat_to_g, mat_to_vc]
        for m in range(3):
            prep_lists = sample_preprocessor(sample_lists + test_lists, use_metrics[m])
            print('Sample preprocessed')
            with open(root + 'pos_5_sep.txt', 'w') as output:
                for n in range(3, 15):
                    print('n = {}'.format(n))
                    accs = [0] * 2
                    for p in range(n_loops):
                        results = hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=n,
                                                      generalized=False, n_fold=10, spec_sens=spec_sens)
                        for q in range(len(results)):
                            accs[q] += results[q]
                    for q in range(len(results)):
                        accs[q] /= n_loops
                    print('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}'.format(n, m, accs[0], accs[1]))
                    output.write('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}\n'.format(n, m, accs[0], accs[1]))

            with open(root + 'pos_5_gen.txt', 'w') as output:
                for n in range(3, 15):
                    accs = [0] * 2
                    for p in range(n_loops):
                        results = hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=n,
                                                      generalized=True, n_fold=10, spec_sens=spec_sens)
                        for q in range(len(results)):
                            accs[q] += results[q]
                    for q in range(len(results)):
                        accs[q] /= n_loops
                    print('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}'.format(n, m, accs[0], accs[1]))
                    output.write('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}\n'.format(n, m, accs[0], accs[1]))


    ################################################################################

    if do_pos_1:
        for k in range(1, 6):
            sample_lists = []
            test_lists = []
            output_sep = open(root + 'pos_1_sep.txt', 'a')
            output_gen = open(root + 'pos_1_gen.txt', 'a')
            output_sep.write('##### Sensor ID = {} #####\n'.format(k))
            output_gen.write('##### Sensor ID = {} #####\n'.format(k))
            arg_dict = {
                "sensor_id": 1, # 1 to 5
                "subject_id": 0, # 0 to 3
                "label_id": 1, # 1 to 5
                "freq": 12.5,
                "db_name": "test_data_fall_1",
            }

            index = 0
            # Four-direction falls (j), five trials each (i)
            for j in range(0, 4):
                sample_lists.append([])
                for i in range(1, 6):
                    # for k in range(1, 6):
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
                            sample_lists[j].append(adding_sample)
                    # index += 1

            # print(len(sample_lists))

            print([len(e) for e in sample_lists])

            arg_dict2 = {
                "sensor_id": 1, # 1 to 5
                "subject_id": 1, # 1 to 5
                "label_id": 1, # 1 to 8
                "freq": 12.5,
                "db_name": "test_data_stage_1",
            }

            # Eight ADLs (i), five subjects each (j)
            for i in range(1, 9):
                test_lists.append([])
                for j in range(1, 6):
                    # for k in range(1, 6):
                        arg_dict2["sensor_id"] = k
                        arg_dict2["label_id"] = i
                        arg_dict2["subject_id"] = j
                        sample = utilities.read_data_from_db(cur, **arg_dict2)
                        # sample = utilities.mat_to_g(sample).tolist()
                        # sample = discretize(sample)
                        if len(sample) != 0:
                            test_lists[i-1].append(sample)

            print([len(e) for e in test_lists])

            n_loops = 5
            use_metrics = [None, mat_to_g, mat_to_vc]
            for m in range(0, 3):
                prep_lists = sample_preprocessor(sample_lists + test_lists, use_metrics[m])

                for n in range(3, 15):
                    accs = [0] * 2
                    for p in range(n_loops):
                        try:
                            results = hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=n,
                                                          generalized=False, n_fold=5, spec_sens=spec_sens)
                        except Exception:
                            results = hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=n,
                                                          generalized=False, n_fold=3, spec_sens=spec_sens)
                        for q in range(len(results)):
                            accs[q] += results[q]
                    for q in range(len(results)):
                        accs[q] /= n_loops
                    print('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}'.format(n, m, accs[0], accs[1]))
                    output_sep.write('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}\n'.format(n, m, accs[0], accs[1]))
                    output_sep.flush()

                # print ([len(everything) for everything in prep_lists])
                # for a in range(len(prep_lists)):
                #     for b in range(len(prep_lists[a])):
                #         print('a = {}, b = {}, first fifteen: {}'.format(a, b, prep_lists[a][b][:15]))

                for n in range(3, 15):
                    accs = [0] * 2
                    for p in range(n_loops):
                        try:
                            results = hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=n,
                                                          generalized=True, n_fold=5, spec_sens=spec_sens)
                        except Exception:
                            print('n_fold = 5 failed, trying n_fold = 3...')
                            results = hmm_model_evaluator(prep_lists, pos_model_indices=[8, 9, 10, 11], n_states=n,
                                                          generalized=True, n_fold=3, spec_sens=spec_sens)
                        for q in range(len(results)):
                            accs[q] += results[q]
                    for q in range(len(results)):
                        accs[q] /= n_loops
                    print('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}'.format(n, m, accs[0], accs[1]))
                    output_gen.write('n = {}, metric = {}, [Type 0]: {:.4g}, [Type 1]: {:.4g}\n'.format(n, m, accs[0], accs[1]))
                    output_gen.flush()


    exit(0)











    # sample_lists = sample_lists[:-8]
    x = np.matrix(np.concatenate([f for f in sample_lists])).T
    l = [len(s) for s in sample_lists]
    # model = hmm.GaussianHMM(n_components=3, n_iter=100)
    model = hmm.GaussianHMM(n_components=3)
    # model.startprob_ = np.array([1, 0, 0])
    # model.transmat_ = np.array([[0.9, 0.1, 0.0],
    #                             [0.0, 0.9, 0.1],
    #                             [0.1, 0.0, 0.9]])
    # model.emissionprob_ = np.array([[1/6,] * 6, [1/6,] * 6, [1/6, ] * 6])
    model.fit(x, l)
    print(model.transmat_)
    print(model.means_)
    print(model.covars_)
    # exit(0)

    models = []
    for i in range(len(test_lists)):
        print("len({}) = {}".format(i, len(test_lists[i])))
        xs = np.matrix(np.concatenate([f for f in test_lists[i]])).T
        ls = [len(s) for s in test_lists[i]]
        ms = hmm.GaussianHMM(n_components=3)
        # a = np.matrix(test_lists[0][0]).T
        ms.fit(xs, ls)
        models.append(ms)

    models.append(model)  # add the fall model to the models
    # exit(1)

    # r = []
    # for s in test_lists:
    #     r.append(model_eval(s, model, models))
    #
    # print(r)
    #
    # r2 = []
    # for s in sample_lists:
    #     r2.append(model_eval(s, model, models))
    #
    # print(r2)
    # {

    # (x, l) = model_samples_generator(test_lists)
    # print(model.score(x, l))

    result = []
    # scores = []
    # max_scores = []

    for s in sample_lists:
        result.append(hmm_classifier(models, s))
    print(result)
    count = 0
    for c in result:
        if c == 8:
            count += 1
    print(count / len(result))

    # for i in range(len(result)):
    #     # print(scores[i])
    #     print(result[i])
    #     print(max_scores[i])
    # # }
    #
    # result = []
    # scores = []
    # max_scores = []

    result2 = []
    for g in test_lists:
        for s in g:
        # print(s)
            result2.append(hmm_classifier(models, s))

    print(result2)
    count = 0
    for c in result2:
        if c != 8:
            count += 1
    print(count / len(result2))
    # for i in range(len(result)):
    #     # print(scores[i])
    #     print(result[i])
    #     print(max_scores[i])


