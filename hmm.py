__author__ = 'Shuo Yu'
from sklearn.externals import joblib
from hmmlearn import hmm
import numpy as np
import utilities
import feature_gen

N = 50

def matching_degree(hmm_model, sample, n=20):
    """
    Accepts an HMM model and a sample, uses a sliding window to figure out the maximum similarity (likelihood)
    between the model and the sample.
    :param hmm_model:
    :param sample:
    :param n:
    :return: maximum likelihood
    """
    sample_set = [sample[x:x+n+1] for x in range(len(sample)+1-n)]
    result_set = []
    for s in sample_set:
        x = hmm_model.score(np.transpose(np.matrix(s)))
        result_set.append(x)
    # print(result_set)
    # print(np.amax(result_set))
    return np.amax(result_set)


def model_samples_generator(sample_lists):
    """
    Accepts a 2D list, generates a tuple (x, l) that contains multiple samples for applying the HMM model
    :param sample: [[1, 2, 3, ...], [4, 5, 6, ...]]
    :return: (a 2D np matrix, [len()])
    """
    x = np.transpose(np.matrix(np.concatenate([s for s in sample_lists])))
    l = [len(s) for s in sample_lists]
    return (x, l)


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

def model_eval(orig_sample, fall_model, adl_models):
    if not isinstance(orig_sample, np.matrix):
        sample = np.matrix(orig_sample).T
    else:
        sample = orig_sample
    scores = []
    scores.append(fall_model.score(sample))
    for adl_model in adl_models:
        scores.append(adl_model.score(sample))
    return np.argmax(scores)


if __name__ == '__main__':
    cur = utilities.db_connect()
    sample_lists = []
    test_lists = []

    start_indices = [20, 18, 18, 17, 19, 18, 17, 18, 19, 17, 17, 18, 16, 15, 16, 14, 17, 14, 11, 9]

    arg_dict = {
        "sensor_id": 1, # 1 to 5
        "subject_id": 0, # 0 to 3
        "label_id": 1, # 1 to 5
        "freq": 12.5,
        "db_name": "test_data_fall_1",
    }

    index = 0
    # Four-direction falls (j), five trials each (i)
    for i in range(1, 6):
        for j in range(0, 4):
            arg_dict["label_id"] = i
            arg_dict["subject_id"] = j
            sample = utilities.read_data_from_db(cur, **arg_dict)
            sample = feature_gen.sample_around_peak(np.matrix(sample), 25, 25)
            sample = utilities.mat_to_g(sample).tolist()
            sample = discretize(sample)
            print(sample)
            # if arg_dict["label_id"] <= 3:
            #     sample = sample[:-60]
            sample_lists.append(sample[start_indices[index]-2:])
            index += 1

    arg_dict2 = {
        "sensor_id": 1, # 1 to 5
        "subject_id": 1, # 1 to 5
        "label_id": 1, # 1 to 8
        "freq": 12.5,
        "db_name": "test_data_stage_1",
    }

    # Eight ADLs (i), five subjects each (j)
    for i in range(1, 9):
        # test_lists.append([])
        for j in range(1, 6):
            arg_dict2["label_id"] = i
            arg_dict2["subject_id"] = j
            sample = utilities.read_data_from_db(cur, **arg_dict2)
            sample = utilities.mat_to_g(sample).tolist()
            sample = discretize(sample)
            if len(sample) != 0:
                test_lists.append(sample)

    # sample_lists = sample_lists[:-8]
    x = np.matrix(np.concatenate([f for f in sample_lists])).T
    l = [len(s) for s in sample_lists]
    # model = hmm.GaussianHMM(n_components=3, n_iter=100)
    model = hmm.MultinomialHMM(n_components=3, n_iter=100, params="ste", init_params="ste")
    # model.startprob_ = np.array([1, 0, 0])
    # model.transmat_ = np.array([[0.9, 0.1, 0.0],
    #                             [0.0, 0.9, 0.1],
    #                             [0.1, 0.0, 0.9]])
    model.fit(x, l)

    models = []
    # for i in range(len(test_lists)):
    #     print("len {}: {}".format(i, len(test_lists[i])))
    #     xs = np.matrix(np.concatenate([f for f in test_lists[i]])).T
    #     ls = [len(s) for s in test_lists[i]]
    #     ms = hmm.MultinomialHMM(n_components=3, n_iter=100, params="ste", init_params="ste")
    #     a = np.matrix(test_lists[0][0]).T
    #     ms.fit(np.matrix(a))
    #     models.append(ms)
    # model.emissionprob_ = np.array([[1/6,] * 6, [1/6,] * 6, [1/6, ] * 6])

    print("Converged:", model.monitor_.converged)
    print(model.transmat_)
    print(model.emissionprob_)
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
    result = []
    scores = []
    max_scores = []
    # (x, l) = model_samples_generator(test_lists)
    # print(model.score(x, l))

    for s in test_lists:
        # print(s)
        model_x = np.transpose(np.matrix(s))
        result.append(model.predict(model_x))
        scores.append(model.score(model_x))
        max_scores.append(matching_degree(model, s))

    for i in range(len(result)):
        # print(scores[i])
        print(result[i])
        print(max_scores[i])

    result = []
    scores = []
    max_scores = []

    for s in sample_lists:
        print(s)
        model_x = np.transpose(np.matrix(s))
        result.append(model.predict(model_x))
        scores.append(model.score(model_x))
        max_scores.append(matching_degree(model, s))

    for i in range(len(result)):
        # print(scores[i])
        print(result[i])
        print(max_scores[i])
    # }
