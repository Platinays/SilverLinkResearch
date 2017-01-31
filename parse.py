__author__ = 'Shuo Yu'

import pymysql
import glob
import h5py
import numpy as np

def db_connect():
    return pymysql.connect(host="127.0.0.1",
                           user="shuoyu",
                           passwd="qoowpyep",
                           db="silverlink",
                           charset='utf8',
                           autocommit=True).cursor()


dict_name_id = {

}

def db_write(cur, ins):
    sql = '''
        INSERT INTO test_data_stage_1b (sensor_id, subject_id, label_id, freq, timestamp, x_accel, y_accel, z_accel)
        VALUES ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')
        '''

    # 1457307771365,null,0,E8BD107D58B4,25.0,-119,163,990
    # print(ins)
    sensor_id = ins[3]
    subject_id = ins[1][-1:]
    # subject_id = int(ins[1])
    label_id = int(ins[2])
    # if label_id == 6 and (subject_id == 2 or subject_id == 4):
    #     return
    # elif label_id == 7:
    #     label_id == 6
    freq = ins[4]
    timestamp = int(ins[0])
    x_accel = ins[5]
    y_accel = ins[6]
    z_accel = ins[7]
    while True:
        try:
            cur.execute(sql % (sensor_id, subject_id, label_id, freq, timestamp, x_accel, y_accel, z_accel))
            break
        except Exception as e:
            print(e)
            timestamp += 1


def csv_to_db_for_fall_1(pattern, cur):
    for file in glob.glob(pattern):
    # file = 'C:/_test_space/6_tests_Shuo_Yu_12.5.csv'
        with open(file, 'r') as fh:
            print('Current file: %s' % file)
            for line in fh:
                if len(line) > 80:  # \n async issue
                    print('async issue for %s' % line)
                    ins_1 = line.split(',')[:8]
                    ins_1[7] = int(ins_1[7]) // 1e13
                    ins_2 = line.split(',')[7:]
                    ins_2[0] = int(ins_2[0]) % 1e13
                    db_write(cur, ins_1)
                    db_write(cur, ins_2)
                else:
                    if len(line) < 10:
                        continue
                    else:
                        if len(line.split(',')) >= 9:
                            temp = line.split(',')
                            temp.pop(1)
                            db_write(cur, temp)


def csv_to_db(pattern, cur):
    for file in glob.glob(pattern):
    # file = 'C:/_test_space/2016-4-23_14_2_23.csv'
        with open(file, 'r') as fh:
            print('Current file: %s' % file)
            for line in fh:
                if len(line) > 80:  # \n async issue
                    print('async issue for %s' % line)
                    ins_1 = line.split(',')[:8]
                    ins_1[7] = int(ins_1[7]) // 1e13
                    ins_2 = line.split(',')[7:]
                    ins_2[0] = int(ins_2[0]) % 1e13
                    db_write(cur, ins_1)
                    db_write(cur, ins_2)
                else:
                    if len(line) < 10:
                        continue
                    else:
                        temp = line.split(',')
                        if temp[1] != 'null':
                            db_write(cur, temp)


def matlab_to_db(pattern, cur):
    sql = '''
        INSERT INTO test_data_farseeing (subject_id, label_id, timestamp, x_accel, y_accel, z_accel)
        VALUES ('%s', '%s', '%s', '%s', '%s', '%s')
        '''
    # label_id refers to is_fall in the mat file

    subject_id = 0
    for file in glob.glob(pattern):
        subject_id += 1
        print('%s: %s' % (subject_id, file))
        d = h5py.File(file)
        rows = np.matrix(d['tmp']).T[:, [0, 2, 3, 4, -1]].tolist()
        for row in rows:
            label_id = row[-1]
            x_accel = round(float(row[1]) * 100)
            y_accel = round(float(row[2]) * 100)
            z_accel = round(float(row[3]) * 100)
            timestamp = round(row[0] * 1000)
            try:
                cur.execute(sql % (subject_id, label_id, timestamp, x_accel, y_accel, z_accel))
            except Exception as e:
                print(e)


if __name__ == '__main__':
    cur = db_connect()
    # matlab_to_db('C:/_test_space/Fall examples/Signal files/*.mat', cur)
    csv_to_db('C:/_test_space/new_samples_0115/2017-1-15_21_15_20.csv', cur)