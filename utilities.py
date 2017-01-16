__author__ = 'Shuo Yu'

import numpy as np
import matplotlib.pyplot as plt
import pymysql
import feature_gen

dict_sensor_id = {
    5: 'E8BD107D58B4',
    4: 'EE9F6185DA6C',
    2: 'EB0BE26E8C52',
    3: 'F7FCFFD2F166',
    1: 'D202B31CD2C3',
}


def db_connect():
    return pymysql.connect(host="127.0.0.1",
                           user="shuoyu",
                           passwd="qoowpyep",
                           db="silverlink",
                           charset='utf8',
                           autocommit=True).cursor()


def write_csv_file(accel_mat, csv_file):
    accel_list = accel_mat.tolist()
    with open(csv_file, 'w') as csv:
        for row in accel_list:
            csv.write(('{:f},' * len(row)).format(*row))
            csv.write('\n')


def read_data_from_db(cur, sensor_id, subject_id, label_id, freq, db_name='test_data_stage_3'):
    sql = '''
        SELECT timestamp, x_accel, y_accel, z_accel
        FROM %s
        WHERE sensor_id = '%s' AND subject_id = %d AND label_id = %d AND freq = %f
        ORDER BY timestamp
    ''' % (db_name, dict_sensor_id[sensor_id], subject_id, label_id, freq)
    cur.execute(sql)
    ret_list = []
    for row in cur:
        ret_list.append([int(row[1]), int(row[2]), int(row[3])])
    return ret_list


def read_data_from_fall_db(cur, sensor_id, subject_id, label_id, freq):
    sql = '''
        SELECT timestamp, x_accel, y_accel, z_accel
        FROM test_data_fall_1
        WHERE sensor_id = '%s' AND subject_id = %d AND label_id = %d AND freq = %f
        ORDER BY timestamp
    ''' % (dict_sensor_id[sensor_id], subject_id, label_id, freq)
    cur.execute(sql)
    ret_list = []
    for row in cur:
        ret_list.append([int(row[1]), int(row[2]), int(row[3])])
    return ret_list


def read_data_from_farseeing_db(cur, subject_id, is_find_falls, ts_start, ts_end):
    if is_find_falls:
        sql = '''
            SELECT timestamp, x_accel, y_accel, z_accel
            FROM test_data_farseeing
            WHERE subject_id = '%s' AND label_id != 0 AND timestamp >= '%s' AND timestamp <= '%s'
        ''' % (subject_id, ts_start, ts_end)
    else:
        sql = '''
            SELECT timestamp, x_accel, y_accel, z_accel
            FROM test_data_farseeing
            WHERE subject_id = '%s' AND timestamp >= '%s' AND timestamp <= '%s'
        ''' % (subject_id, ts_start, ts_end)
    cur.execute(sql)
    ret_list = []
    for row in cur:
        ret_list.append([int(row[1]), int(row[2]), int(row[3])])
    return ret_list


def db_to_csv(csv_name, db_name, sensor_id, subject_id, label_id):
    cur = db_connect()
    mat = np.matrix(read_data_from_db(cur, sensor_id, subject_id, label_id, 12.5, db_name))
    write_csv_file(mat, csv_name)


def gen_x_axis(value_list, freq=12.5):
    return [x / freq for x in range(len(value_list))]


def gen_plot_title(sensor_id, subject_id, label_id, freq, db_name):
    pos_dict = {
        1: "Necklace",
        2: "Waist",
        3: "Chest",
        4: "Right Shoe",
        5: "Left Shoe",
    }
    label_dict = {
        1: "Sit to Stand (1)",
        2: "Quiet Stance (2)",
        3: "10 Meters Walking (3)",
        4: "10 Meters Walking (4)",
        5: "Stand to Sit (5)",
        6: "Timed Up & Go (6)",
    }
    db_dict = {
        "test_data_stage_3": "Revised Experiment",
        "test_data_fall_1": "Simulated Fall",
    }

    return "Acceleration Signals for {} Sensor during {} Test, Subject {} in {} with Freq {}".format(
        pos_dict[sensor_id], label_dict[label_id], subject_id, db_dict[db_name], freq)


def mat_to_g(value_list):
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    mat = np.linalg.norm(mat, ord=2, axis=1)
    mat = np.transpose(mat)
    return mat


def mat_to_vc(value_list):
    if not isinstance(value_list, np.matrix):
        mat = np.matrix(value_list)
    else:
        mat = value_list
    mat = feature_gen.calc_vt_comp_with_rem_mag(value_list)
    mat = np.transpose(mat)
    return mat


def plot_x_y(value_list, title='', option='xyz', jerk=False):
    if isinstance(value_list, np.matrix):
        if option == 'xyz':
            mat = np.transpose(value_list).tolist()
            print(mat)
            x_list = gen_x_axis(mat[0])
            plt.plot(x_list, mat[0], 'r-', x_list, mat[1], 'g-', x_list, mat[2], 'b-')
        elif option == 'g':
            if jerk == False:
                mat = mat_to_g(value_list).tolist()
                x_list = gen_x_axis(mat)
                plt.axis([0, x_list[-1], 0, 4000])
            else:
                mat = np.diff(mat_to_g(value_list)).tolist()
                x_list = gen_x_axis(mat)
                plt.axis([0, x_list[-1], -4000, 4000])
            print(mat)
            plt.plot(x_list, mat, 'b-')
        elif option == 'vc':
            mat = mat_to_vc(value_list).tolist()
            x_list = gen_x_axis(mat[0])
            plt.plot(x_list, mat[0], 'b-', x_list, mat[1], 'r-')
            plt.axis([0, x_list[-1], -4000, 4000])
    else:
        x_list = gen_x_axis(value_list)
        plt.plot(x_list, value_list)
        plt.axis([0, x_list[-1], -2000, 2000])
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(r'Acceleration ($10^{-3}g$)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    cur = db_connect()
    csv_file = 'c:/1.csv'
    arg_dict = {
        "sensor_id": 5,
        "subject_id": 1,
        "label_id": 5,
        "freq": 12.5,
        "db_name": "test_data_fall_1",
    }
    mat = np.matrix(read_data_from_db(cur, **arg_dict))
    plot_x_y(mat, title=gen_plot_title(**arg_dict), option='vc', jerk=False)