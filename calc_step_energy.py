__author__ = 'Shuo Yu'

# import numpy
import math
import csv
import rethinkdb as r


def moving_average(num_list, k=8):
    ret_list = []
    c = 0
    sum = 0
    for i in range(len(num_list)):
        sum += num_list[i]
        c += 1
        if c > k:
            sum -= num_list[i - k]
            c -= 1
        ret_list.append(sum / c)
    return ret_list


def get_sign_list(g_list, k=1, thres=0.05):
    """
    Convert a list of g-force to a list of signs \in {1, 0, -1} based on their values compared to the average.
    :param g_list: a list of g-force accel.
    :param thres: g-force is converted to sign 0 if \in [avg + thres * avg, avg - thres * avg]
    :return: a list of signs \in {1, 0, -1}
    """
    g_list = moving_average(g_list, k)
    avg = sum(g_list) / len(g_list)
    sign_list = []
    for i in range(len(g_list)):
        if (g_list[i] - avg) > thres * avg:
            sign_list.append(1)
        elif (g_list[i] - avg) < -thres * avg:
            sign_list.append(-1)
        else:
            sign_list.append(0)
    return sign_list


def get_step(g_list, thres=0.05):
    """
    Get the step count based on a series of g-force data.
    :param g_list: a list of g-force accel.
    :param thres: controls the sensitivity of step detection.
    :return: step count
    """

    sign_list = get_sign_list(g_list, thres)
    # print(g_list)
    step_count = 0
    prev_state = 0
    for cur_state in sign_list:
        if prev_state == 1 and cur_state == -1:
            step_count += 1
            prev_state = -1
        elif prev_state == -1 and cur_state == 1:
            step_count += 1
            prev_state = 1
        elif prev_state == 0:
            prev_state = cur_state
    return step_count // 2


def mean_crossing_rate(g_list, thres=0.05):
    """
    Get the step count based on a series of g-force data.
    :param g_list: a list of g-force accel.
    :param thres: controls the sensitivity of step detection.
    :return: step count
    """

    sign_list = get_sign_list(g_list, k=1, thres=thres)
    # print(g_list)
    step_count = 0
    prev_state = 0
    for cur_state in sign_list:
        if prev_state == 1 and cur_state == -1:
            step_count += 1
            prev_state = -1
        elif prev_state == -1 and cur_state == 1:
            step_count += 1
            prev_state = 1
        elif prev_state == 0:
            prev_state = cur_state
    return step_count / (len(g_list) - 1)


def get_energy_by_step(step_count):
    """Return calories calculation based on an estimation formula:
    calories = 0.045 * steps
    """
    return 0.045 * step_count


def load_accel_from_db(sensor_id='C71C990F9D00', start_ts=1446716890, end_ts=1446716895,
                       ret_ts=False, ret_xyz=False, ret_g=True):
    """
    Read the database for a specific sensor (sensor_id) during a specific time period (start_ts to end_ts, exclusive).
    :param sensor_id: sensor id
    :param start_ts: start timestamp
    :param end_ts: end timestamp
    :param ret_ts: set to True to return timestamps.
    :param ret_xyz: set to True to return tri-axial accel.
    :param ret_g: set to True to return g-force accel.
    :return: a list of lists for timestamps, tri-axial accel. and/or g-force accel.
    """
    conn = r.connect(host='104.236.129.116', db='test', auth_key='AIlab123')
    cur = r.table('sensor_logs').order_by(index='t').filter(lambda data:
        (data['s'] == sensor_id) & (data['t'] > start_ts) & (data['t'] < end_ts)
    ).run(conn)
    results = list(cur)

    ts = []
    accel_x = []
    accel_y = []
    accel_z = []
    accel_g = []
    for item in results:
        t = float(item['t'])
        x = float(item['x'])
        y = float(item['y'])
        z = float(item['z'])
        g = math.sqrt(x ** 2 + y ** 2 + z ** 2)
        ts.append(t)
        accel_x.append(x)
        accel_y.append(y)
        accel_z.append(z)
        accel_g.append(g)
    ret = []
    if ret_ts:
        ret.append(ts)
    if ret_xyz:
        ret.append(accel_x)
        ret.append(accel_y)
        ret.append(accel_z)
    if ret_g:
        ret.append(accel_g)
    return ret

def load_accel_from_csv(path='c:/test.csv', ret_ts=False, ret_xyz=False, ret_g=True):
    """
    Read a CSV file from path. Each line of the CSV file consists 4 numbers: timestamp, x, y, and z-accel.
    :param path: the path for the CSV file.
    :param ret_ts: set to True to return timestamps.
    :param ret_xyz: set to True to return tri-axial accel.
    :param ret_g: set to True to return g-force accel.
    :return: a list of lists for timestamps, tri-axial accel. and/or g-force accel.
    """

    ts = []
    accel_x = []
    accel_y = []
    accel_z = []
    accel_g = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            t = float(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            g = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            ts.append(t)
            accel_x.append(x)
            accel_y.append(y)
            accel_z.append(z)
            accel_g.append(g)
    ret = []
    if ret_ts:
        ret.append(ts)
    if ret_xyz:
        ret.append(accel_x)
        ret.append(accel_y)
        ret.append(accel_z)
    if ret_g:
        ret.append(accel_g)
    return ret


def get_step_energy_from_db(sensor_id='C71C990F9D00', start_ts=1446716890, end_ts=1446716895,
                            thres=0.05, height=170, weight=60):
    """
    Calculate step counts and energy consumption for a specific sensor during a specific period of time.
    :param sensor_id: sensor id
    :param start_ts: start timestamp
    :param end_ts: end timestamp
    :param thres: controls the sensitivity of step counts, 0.05 by default
    :param height: user height, not used for now
    :param weight: user weight, not used for now
    :return: a tuple: (step_count, energy_consumption). energy_consumption is in calories.
    """
    g_list = load_accel_from_db(sensor_id, start_ts, end_ts)[0]
    step_count = get_step(g_list, thres)
    energy_consumption = get_energy_by_step(step_count)
    return step_count, energy_consumption


def get_step_energy_from_csv(path = 'c:/test.csv'):
    g_list = load_accel_from_csv(path)[0]
    step_count = get_step(g_list)
    energy_consumption = get_energy_by_step(step_count)
    return step_count, energy_consumption


def get_activity_count(ts_list, interval=60):
    """
    Receive a list of ORDERED (ascending) timestamps. Return the activity count based on the given interval.
    Two data points within an interval will be treated as the same activity.
    :param ts_list: a list of ORDERED timestamps
    :param interval: activity time interval in SECONDS, 60 by default
    :return: an integer of activity counts
    """
    if len(ts_list) == 0:
        return 0

    count = 0
    prev_ts = ts_list[0]
    cur_ts = 0
    for i in range(len(ts_list)):
        cur_ts = ts_list[i]
        if cur_ts - prev_ts >= interval:
            count += 1
        prev_ts = cur_ts
    return count


def get_active_duration(ts_list, time_window=60):
    """
    Receive a list of ORDERED (ascending) timestamps. Return the active duration based on the given time window.
    Whenever there is a data point inside the time window, the entire time window is marked as "active".
    :param ts_list: a list of ORDERED timestamps
    :param time_window: the time window in SECONDS, 60 by default
    :return: an integer of active duration in SECONDS (if time_window is float, the return value is float)
    """
    if len(ts_list) == 0:
        return 0

    count = 0
    prev_flag_ts = 0
    cur_flag_ts = 0
    for i in range(len(ts_list)):
        cur_ts = ts_list[i]
        cur_flag_ts = cur_ts // time_window
        if cur_flag_ts != prev_flag_ts:
            count += 1
            prev_flag_ts = cur_flag_ts
    return count * time_window


def get_activity_count_from_db(sensor_id='C71C990F9D00', start_ts=1446716890, end_ts=1446716895, interval=60):
    ts_list = load_accel_from_db(sensor_id, start_ts, end_ts,
                                                           ret_ts=True, ret_xyz=False, ret_g=False)[0]
    count = get_activity_count(ts_list, interval)
    return count


def get_active_duration_from_db(sensor_id='C71C990F9D00', start_ts=1446716890, end_ts=1446716895, time_window=60):
    ts_list = load_accel_from_db(sensor_id, start_ts, end_ts,
                                                           ret_ts=True, ret_xyz=False, ret_g=False)[0]
    duration = get_active_duration(ts_list, time_window)
    return duration


if __name__ == '__main__':
    # print(get_activity_count_from_db())
    f = open('c:/metawear.txt')
    read_list = []
    for line in f:
        # print(float(line))
        read_list.append(float(line))
    print(get_step(read_list))