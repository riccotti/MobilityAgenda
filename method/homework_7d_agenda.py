from method.homework_agenda import *

from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def build_homework_7d(train, history_order_dict, grid_length, time_lenght, min_lon, min_lat, tzoffset):

    weekday_train = {
        0: {'trajectories': dict()},
        1: {'trajectories': dict()},
        2: {'trajectories': dict()},
        3: {'trajectories': dict()},
        4: {'trajectories': dict()},
        5: {'trajectories': dict()},
        6: {'trajectories': dict()},
    }

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        if tid not in train['trajectories']:
            continue
        traj = train['trajectories'][tid]
        ts = datetime.datetime.fromtimestamp(traj.start_point()[2] / 1000 + tzoffset * 3600)
        weekday = ts.weekday()
        weekday_train[weekday]['trajectories'][tid] = traj

    weekday_model = dict()
    for weekday in weekday_train:
        train_wd = weekday_train[weekday]
        homework_model = build_homework(train_wd, history_order_dict, grid_length, time_lenght, min_lon, min_lat,
                                        tzoffset)
        weekday_model[weekday] = homework_model

    return weekday_model


def generate_homework_7d_agenda(weekday, homework_7d_model, grid_length, time_lenght, min_lon, min_lat,
                                time_start=0, time_offset=300, time_stop=86400):

    homework_model = homework_7d_model[weekday]
    if homework_model is None:
        if len(homework_7d_model) > 0:
            closest_weekday = np.min([abs(x - weekday) for x in homework_7d_model]) + weekday
            if closest_weekday in homework_7d_model:
                homework_model = homework_7d_model[closest_weekday]
            else:
                homework_model = homework_7d_model[homework_7d_model.keys()[0]]

    if homework_model is None:
        return None

    agenda = generate_homework_agenda(homework_model, grid_length, time_lenght, min_lon, min_lat,
                                      time_start=time_start, time_offset=time_offset, time_stop=time_stop)

    return agenda


def main():

    params = rome_params

    con = get_connection()
    cur = con.cursor()

    input_table = params['input_table']
    min_lat = params['min_lat']
    min_lon = params['min_lon']
    tzoffset = params['tzoffset']

    uid = 146099 # rome
    #uid = 659447 # london
    week_limit = 42

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)

    grid_length = 1000
    time_lenght = 3600
    weekday = 6

    homework_7d_model = build_homework_7d(train, history_order_dict, grid_length, time_lenght,
                                          min_lon, min_lat, tzoffset)

    homework_7d_agenda = generate_homework_7d_agenda(weekday, homework_7d_model, grid_length, time_lenght,
                                                     min_lon, min_lat, time_start=0, time_offset=300, time_stop=86400)

    print '------'

    for event_time in sorted(homework_7d_agenda):
        event = homework_7d_agenda[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]


if __name__ == "__main__":
    main()
