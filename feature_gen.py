__author__ = 'Shuo Yu'

import numpy as np
import math
from dtw import dtw
from utilities import *
from calc_step_energy import moving_average
from calc_step_energy import mean_crossing_rate


def mag(num_list):
    sq_sum = 0
    for n in num_list:
        sq_sum += n ** 2
    return sq_sum ** 0.5


def gen_rot_mat(deg):
    rad = deg * math.pi / 180
    rot_mat = np.dot(
            np.matrix(
            [[math.cos(rad), math.sin(rad), 0],
             [-math.sin(rad), math.cos(rad), 0],
             [0, 0, 1]]
        ),
        np.matrix(
            [[1, 0, 0],
             [0, math.cos(2 * rad), math.sin(2 * rad)],
             [0, -math.sin(2 * rad), math.cos(2 * rad)]
            ]
        )
    )
    return rot_mat


def gen_rot_deg(subject_id, label_id):
    return (subject_id * 7 + label_id) * 11 * 3


def apply_rot_mat(accel_mat, deg):
    return np.dot(accel_mat, gen_rot_mat(deg))


def calc_magnitude(accel_mat, n=1):
    '''

    :param accel_mat: 2D MATRIX, a series of 3-axial accel data, [[0, 1, 0], [1, 0, 0], ...]
    :return: a vector of magnitude
    '''
    ret_vec = []
    if isinstance(accel_mat, np.matrix):
        lists = accel_mat.tolist()
    else:
        lists = accel_mat

    for row in lists:
        ret_vec.append(mag(row))
    if n == 1:
        return np.array(ret_vec)
    else:
        return np.array(moving_average(ret_vec, n))


def gen_feature_vector(accel_col):
    ret_list = []
    # ret_list.append(np.amin(np.absolute(accel_col)))
    ret_list.append(np.amax(np.absolute(accel_col)))
    ret_list.append(np.mean(np.absolute(accel_col)))
    ret_list.append(np.var(accel_col) ** 0.5)
    ret_list.append(np.subtract(*np.percentile(accel_col, [75, 25])))
    ret_list.append(mean_crossing_rate(accel_col.reshape(-1).tolist()[0]))
    return ret_list


def gen_feature_vector_list(accel_mat_list, calib=0):
    '''
    Feature list. [min, max, abs_mean, variance, ,
    :param accel_mat_list:
    :return:
    '''
    feature_vector_list = []
    for accel_mat in accel_mat_list:
        proc_mat = calibrate(accel_mat, calib)
        feature_vector = []
        r, c = proc_mat.shape
        for i in range(c):
            accel_col = proc_mat[:, i]
            feature_vector += gen_feature_vector(accel_col)
        feature_vector_list.append(feature_vector)
    return feature_vector_list


def write_arff_file(feature_vector_list, output_file):
    with open(output_file, 'w') as arff:
        arff.write('@RELATION fall_detecion\n')
        num_set = len(feature_vector_list[0]) // 6
        for i in range(num_set):
            arff.write('''@ATTRIBUTE min_{0} NUMERIC
@ATTRIBUTE max_{0} NUMERIC
@ATTRIBUTE mean_{0} NUMERIC
@ATTRIBUTE var_{0} NUMERIC
@ATTRIBUTE iqr_{0} NUMERIC
@ATTRIBUTE zero_crossing_rate_{0} NUMERIC
'''.format(i))

        arff.write('@ATTRIBUTE class_label {fall, non_fall}\n')
        arff.write('@DATA\n')
        for i in range(len(feature_vector_list)):
            feature_vector = feature_vector_list[i]
            arff.write(('{:f},' * len(feature_vector)).format(*feature_vector))
            if i < 20:
                arff.write('fall')
            else:
                arff.write('non_fall')
            arff.write('\n')


def identify_peak(accel_mat, n=1):
    '''

    :param accel_mat: 2D MATRIX, a series of 3-axial accel data, [[0, 1, 0], [1, 0, 0], ...]
    :return: the magnitude and index of the peak in terms of magnitude
    '''
    mag_vec = calc_magnitude(accel_mat, n)
    return (np.amax(mag_vec), np.argmax(mag_vec))


def identify_valley(accel_mat, n=1, peak_index=-1):
    '''

    :param accel_mat: 2D MATRIX, a series of 3-axial accel data, [[0, 1, 0], [1, 0, 0], ...]
    :return: the magnitude and index of the valley in terms of magnitude
    '''
    mag_vec = calc_magnitude(accel_mat, n)
    peak_index = identify_peak(accel_mat, n)[1]
    return (np.amin(mag_vec[:peak_index]), np.argmin(mag_vec[:peak_index]))


def sample_around_peak(accel_mat, before=24, after=32):
    '''
    Generate a snippet of acceleration sample around the peak detected
    :param accel_mat:
    :param before:
    :param after:
    :return:
    '''
    peak_index = identify_peak(accel_mat)[1]
    # print('peak_index: {}'.format(peak_index))
    start = peak_index - before if peak_index - before >= 0 else 0
    end = peak_index + after if peak_index + after <= accel_mat.shape[0] else accel_mat.shape[0]
    temp_mat = accel_mat[start:end, :]
    if peak_index - before < 0:
        pre = np.array(accel_mat[0, :].tolist() * (before - peak_index)).reshape(-1, 3)
        temp_mat = np.concatenate((np.matrix(pre), temp_mat))
    if peak_index + after > accel_mat.shape[0]:
        post = np.array(accel_mat[-1, :].tolist() * (peak_index + after - accel_mat.shape[0])).reshape(-1, 3)
        # print(post)
        temp_mat = np.concatenate((temp_mat, np.matrix(post)))

    return temp_mat


def convert_to_magnitude_list(mat_list):
    ret_list = []
    for mat in mat_list:
        ret_list.append(calc_magnitude(mat))
    return ret_list


# def calc_gravity(accel_mat):
#     k = 16
#     if accel_mat.shape[0] < 12:
#         return np.mean(accel_mat, axis=0)
#     else:
#         return np.mean(accel_mat[:12, :], axis=0)


def calc_gravity(accel_mat):
    """
    Use the acceleration vectors (1-sec period) after the shock to estimate an axis
    :param accel_mat:
    :return:
    """
    # p = identify_peak(accel_mat)[1]
    r, c = accel_mat.shape
    p = 0
    # print(p)
    if r > 48:
        return np.mean(accel_mat[p+36:p+48, :], axis=0)
    else:
        return np.mean(accel_mat[p:p+12, :], axis=0)


def calc_vt_comp(accel_mat):
    vt_vec = calc_gravity(accel_mat)
    unit_vt_vec = vt_vec / np.linalg.norm(vt_vec)
    # a_gi = (a_i * g^T) * g
    # ret_mat = np.dot(np.dot(accel_mat, np.transpose(unit_vt_vec)), unit_vt_vec) - vt_vec
    ret_mat = np.dot(np.dot(accel_mat, np.transpose(unit_vt_vec)), unit_vt_vec)
    return ret_mat


def calc_rem_comp_excl_vt(accel_mat):
    ret_mat = accel_mat - calc_vt_comp(accel_mat)
    return ret_mat


def directed_vec_mag(accel_mat, vec):
    ret_list = []
    for i in range(accel_mat.shape[0]):
        a = accel_mat[i, 0] * 1000 / vec[0, 0]
        ret_list.append(a)
    return np.matrix(np.array(ret_list).reshape(-1, 1))


def calc_vt_comp_with_rem_mag(accel_mat):
    vt_vec = calc_gravity(accel_mat)
    unit_vt_vec = vt_vec / np.linalg.norm(vt_vec)
    vt_comp_mat = calc_vt_comp(accel_mat)
    # print('vt_comp_mat:', vt_comp_mat)
    vt_mag_mat = directed_vec_mag(vt_comp_mat, vt_vec)
    rem_mag_mat = np.matrix(calc_magnitude((accel_mat - vt_comp_mat) * 1000 / np.linalg.norm(vt_vec)).reshape(-1, 1))
    return np.concatenate((vt_mag_mat, rem_mag_mat), axis = 1)


def calibrate(accel_mat, calib=1):
    ret_mat = None
    if calib == 1:
        ret_mat = calc_vt_comp_with_rem_mag(accel_mat)
    elif calib == 2:
        ret_mat = np.matrix(calc_magnitude(accel_mat).reshape(-1, 1))
    else:
        ret_mat = accel_mat
    return ret_mat


def calc_euclid_dist(accel_mat_1, accel_mat_2, calib=1, avg=1):
    accel_mat_a = calibrate(accel_mat_1, calib)
    accel_mat_b = calibrate(accel_mat_2, calib)
    ret_mat = np.matrix([])
    num_row, num_col = accel_mat_a.shape
    for j in range(num_col):
        ret_mat = np.concatenate((ret_mat, np.mean(np.absolute(accel_mat_a[:, j] - accel_mat_b[:, j]), axis=0)), axis=1)
    if avg == 1:
        return np.mean(ret_mat, axis=1).tolist()[0][0]
    elif avg == 2:
        # print(ret_mat)
        # print(calc_magnitude(ret_mat))
        return calc_magnitude(ret_mat).tolist()[0]
    else:
        return ret_mat


def calc_dtw_dist(accel_mat_1, accel_mat_2, calib=1, avg=1):
    accel_mat_a = calibrate(accel_mat_1, calib)
    accel_mat_b = calibrate(accel_mat_2, calib)
    ret_mat = np.matrix([])
    num_row, num_col = accel_mat_a.shape
    for j in range(num_col):
        mat = dtw(accel_mat_a[:, j], accel_mat_b[:, j], dist=lambda x, y: abs(x - y))[0] * 2 # The dtw function seems to have some issue in calc the dist in half
        ret_mat = np.concatenate((ret_mat, np.matrix([[mat]])), axis=1)
    if avg == 1:
        return np.mean(ret_mat, axis=1).tolist()[0][0]
    elif avg == 2:
        return calc_magnitude(ret_mat).tolist()[0]
    else:
        return ret_mat


def load_fall_mat_list(sensor_id, rotate=True):
    ret_mat_list = []
    for subject_id in range(0, 4):
        for label_id in range(1, 6):
            new_sample = read_data_from_fall_db(cur, sensor_id=sensor_id, subject_id=subject_id, label_id=label_id, freq=12.5)
            if len(new_sample) > 0:
                if rotate:
                    amat = np.matrix(apply_rot_mat(new_sample, gen_rot_deg(subject_id, label_id)))
                else:
                    amat = np.matrix(new_sample)
                ret_mat_list.append(sample_around_peak(amat))
    return ret_mat_list


def load_adl_mat_list(sensor_id, rotate=True):
    ret_mat_list = []
    for subject_id in range(1, 6):
        for label_id in range(1, 9):
            new_sample = read_data_from_db(cur, sensor_id=sensor_id, subject_id=subject_id, label_id=label_id, freq=12.5)
            if len(new_sample) > 0:
                if rotate:
                    amat = np.matrix(apply_rot_mat(new_sample, gen_rot_deg(subject_id, label_id)))
                else:
                    amat = np.matrix(new_sample)
                # ret_mat_list.append(sample_around_peak(amat))
                ret_mat_list.append(amat)
    return ret_mat_list


def write_sim_to_csv(mat_list_1, mat_list_2, file_suffix='', k=1, calib=1, avg=1):
    num_row = len(mat_list_1)
    num_col = len(mat_list_2)
    result_lists_euclid = np.zeros((num_row, num_col))
    result_lists_dtw = np.zeros((num_row, num_col))
    for i in range(num_row):
        print('i = {}'.format(i))
        for j in range(num_col):
            a = mat_list_1[i]
            b = mat_list_2[j]
            if a.size == 0 or b.size == 0:
                result_lists_euclid[i, j] = -1
                result_lists_dtw[i, j] = -1
            else:
                result_lists_euclid[i, j] = calc_euclid_dist(a, b, calib=calib, avg=avg)
                result_lists_dtw[i, j] = calc_dtw_dist(a, b, calib=calib, avg=avg)

    with open('c:\euclid_dtw\euclid_{}.csv'.format(file_suffix), mode='w') as output:
        # output.write(' ,')
        # for j in range(num_col):
        #     output.write('{},'.format(j))
        # output.write('\n')
        for i in range(num_row):
            # output.write('{},'.format(i))
            for j in range(num_col):
                output.write(str(result_lists_euclid[i, j]) + ',')
            output.write('\n')
    with open('c:\euclid_dtw\dtw_{}.csv'.format(file_suffix), mode='w') as output:
        # output.write(' ,')
        # for j in range(num_col):
        #     output.write('{},'.format(j))
        # output.write('\n')
        for i in range(num_row):
            # output.write('{},'.format(i))
            for j in range(num_col):
                output.write(str(result_lists_dtw[i, j]) + ',')
            output.write('\n')
    r1 = kNN(result_lists_euclid, k)
    r2 = kNN(result_lists_dtw, k)
    print(sum(r1[:20]), sum(r1[20:]))
    print(sum(r2[:20]), sum(r2[20:]))


def kNN(dist_lists, k=1, num_pos_samples=20):
    result_list = []
    for num_row in range(len(dist_lists)):
        row = dist_lists[num_row]
        c = -1
        for i in range(k + 1):
            if num_row < num_pos_samples and np.argmin(row) < num_pos_samples:
                c += 1
            elif num_row >= num_pos_samples and np.argmin(row) >= num_pos_samples:
                c += 1
            row[np.argmin(row)] = 1e5
        result_list.append(c)
    return result_list


def classify_by_shock_thres(accel_mat_list):
    peak_list = []
    true_pos_list = []
    true_neg_list = []
    for accel_mat in accel_mat_list:
        peak_list.append(identify_peak(accel_mat)[0])
    for thres in range(1000, 4100, 100):
        true_pos = 0
        true_neg = 0
        length = len(peak_list)
        for i in range(length):
            if i < 20 and peak_list[i] >= thres:
                true_pos += 1
            if i >= 20 and peak_list[i] < thres:
                true_neg += 1
        true_pos_list.append(true_pos)
        true_neg_list.append(true_neg)
    return [true_pos_list, true_neg_list]


if __name__ == '__main__':
    cur = db_connect()
    option = 1
    sensor_id = 1
    calib = 0
    if option == 1:
        pass
        # for subject_id in range(0, 4):
        #     for label_id in range(1, 6):
        #         amat = np.matrix(read_data_from_fall_db(cur, sensor_id=sensor_id, subject_id=subject_id, label_id=label_id, freq=12.5))
        #         fall_mat_list.append(sample_around_peak(amat))
    elif option == 2:
        for subject_id in range(1, 6):
            for label_id in range(1, 7):
                amat = np.matrix(read_data_from_db(cur, sensor_id=sensor_id, subject_id=subject_id, label_id=label_id, freq=12.5))
                if amat.size == 0:
                    print('{}, {}, {} does not exist!'.format(sensor_id, subject_id, label_id))
                else:
                    print(identify_peak(amat, 1), identify_valley(amat, 1))
    elif option == 3:
        amat = np.matrix(read_data_from_fall_db(cur, sensor_id=sensor_id, subject_id=0, label_id=3, freq=12.5))
        print(sample_around_peak(amat))
    # for i in range(len(fall_mat_list)):
    #     print(mat)
    #     print(calc_gravity(mat))
    #     print(calc_vt_comp_with_rem_mag(mat))

    # b = adl_mat_list[0]
    # norm_a = calc_vt_comp_with_rem_mag(a)
    # norm_b = calc_vt_comp_with_rem_mag(b)
    # print(norm_a)
    # print(norm_b)
    # print(b)
    # for b in fall_mat_list:
    #     gb = calc_gravity(b)
    #     print(np.linalg.norm(gb))
    # print(norm_b)

    # calib == 1: 2-d; calib == 2: 1-d
    # avg == 1: mean; avg == 2: magnitude
    for sensor_id in range(1, 6):
        for calib in range(0, 3):
            fall_mat_list = load_fall_mat_list(sensor_id=sensor_id, rotate=True)
            adl_mat_list = load_adl_mat_list(sensor_id=sensor_id, rotate=True)
            all_mat_list = fall_mat_list + adl_mat_list
            print(len(all_mat_list))
    # write_csv_file(adl_mat_list[0], 'c:/_test_space/stand_sample2.csv')
            write_arff_file(gen_feature_vector_list(all_mat_list, calib=calib), 'c:/_test_space/arff_nomin_{}_{}.arff'.format(sensor_id, calib))
    # print(classify_by_shock_thres(all_mat_list))
    # write_sim_to_csv(all_mat_list, all_mat_list, 'all_all', k=1, calib=0, avg=2)