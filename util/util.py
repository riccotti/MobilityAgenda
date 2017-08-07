import math
import pytz
import datetime


R_EARTH = 6371000


def dist2angle(dist):

    return dist * 180.0 / math.pi / R_EARTH


def get_ordered_history(imh):
    history_order_dict = dict()
    for tid in imh['trajectories']:
        ts = datetime.datetime.fromtimestamp(imh['trajectories'][tid].start_point()[2] / 1000.0)
        history_order_dict[tid] = ts
    return history_order_dict


def train_test_partition_weeklimit(imh, history_order_dict, week_limit):
    train = {'trajectories': dict()}
    test = {'trajectories': dict()}

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        ts = imh['trajectories'][tid].start_point()[2] / 1000
        week = datetime.datetime.fromtimestamp(ts).isocalendar()[1]
        if week <= week_limit:
            train['trajectories'][tid] = imh['trajectories'][tid]
        else:
            test['trajectories'][tid] = imh['trajectories'][tid]

    return train, test


def train_test_partition_date(imh, history_order_dict, date):

    train = {'trajectories': dict()}
    test = {'trajectories': dict()}
    test_days = set()

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        ts = imh['trajectories'][tid].start_point()[2] / 1000
        dt = datetime.datetime.fromtimestamp(ts)
        upper_limit = datetime.datetime.strptime('2015-12-31', '%Y-%m-%d')
        if dt <= date:
            train['trajectories'][tid] = imh['trajectories'][tid]
        elif dt <= upper_limit:
            test['trajectories'][tid] = imh['trajectories'][tid]
            test_days.add(dt.timetuple().tm_yday)

    return train, test, sorted(test_days)


def train_test_partition_date2(imh, history_order_dict, date):

    train = {'trajectories': dict()}
    test = {'trajectories': dict()}
    test_days = set()

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        ts = imh['trajectories'][tid].start_point()[2] / 1000
        dt = datetime.datetime.fromtimestamp(ts)
        upper_limit = datetime.datetime.strptime('2015-12-31', '%Y-%m-%d')
        if dt <= date:
            train['trajectories'][tid] = imh['trajectories'][tid]
        elif dt <= upper_limit:
            test['trajectories'][tid] = imh['trajectories'][tid]
            test_days.add((dt.timetuple().tm_yday, dt.weekday()))

    return train, test, sorted(test_days)


def train_test_partition_growing(imh, history_order_dict, min_nbr_days=14):

    train = {'trajectories': dict()}
    test = {'trajectories': dict()}
    test_days = set()
    train_days = set()

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        ts = imh['trajectories'][tid].start_point()[2] / 1000
        dt = datetime.datetime.fromtimestamp(ts)
        if len(train_days) <= min_nbr_days:
            train['trajectories'][tid] = imh['trajectories'][tid]
            train_days.add((dt.timetuple().tm_yday, dt.weekday()))
        else:
            test['trajectories'][tid] = imh['trajectories'][tid]
            test_days.add((dt.timetuple().tm_yday, dt.weekday()))

    return train, test, sorted(test_days), len(train_days)


def train_test_partition_percentage(imh, history_order_dict, perc=0.7):
    yday_set = set()

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        ts = imh['trajectories'][tid].start_point()[2] / 1000
        yday = datetime.datetime.fromtimestamp(ts).timetuple().tm_yday
        yday_set.add(yday)

    ndays = len(yday_set)
    part_idx = int(ndays * perc)
    part_idx = ndays - 1 if part_idx == ndays else part_idx

    part_day = sorted(yday_set)[part_idx]
    test_days = sorted(yday_set)[part_idx + 1:]

    train = {'trajectories': dict()}
    test = {'trajectories': dict()}

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        ts = imh['trajectories'][tid].start_point()[2] / 1000
        yday = datetime.datetime.fromtimestamp(ts).timetuple().tm_yday
        if yday <= part_day:
            train['trajectories'][tid] = imh['trajectories'][tid]
        else:
            test['trajectories'][tid] = imh['trajectories'][tid]

    return train, test, test_days


rome_params = {
    'input_table': 'agenda.rome1000',
    'min_lat': 41.24,
    'min_lon': 11.59,
    'tzoffset': int(datetime.datetime.now(pytz.timezone('Europe/Rome')).strftime('%z')[:3]),
    'traintest_date': datetime.datetime.strptime('2015-05-03', '%Y-%m-%d'),
}

london_params = {
    'input_table': 'agenda.london1000',
    'min_lat': 51.15,
    'min_lon': -0.89,
    'tzoffset': int(datetime.datetime.now(pytz.timezone('Europe/London')).strftime('%z')[:3]),
    'traintest_date': datetime.datetime.strptime('2015-05-03', '%Y-%m-%d'),
}

boston_params = {
    'input_table': 'agenda.boston1000',
    'min_lat': 40.91,
    'min_lon': -73.98,
    'tzoffset': int(datetime.datetime.now(pytz.timezone('US/Eastern')).strftime('%z')[:3]),
    'traintest_date': datetime.datetime.strptime('2015-05-03', '%Y-%m-%d'),
}

beijing_params = {
    'input_table': 'agenda.beijing',
    'min_lat': 38.945889,
    'min_lon': 114.698094,
    'tzoffset': int(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%z')[:3]),
    'traintest_date': datetime.datetime.strptime('2008-05-04', '%Y-%m-%d'),
}
