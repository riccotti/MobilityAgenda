import numpy as np

from collections import defaultdict

from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def get_lonlat(loc_cell, grid_length2, min_lon, min_lat):
    lon = loc_cell[0] * grid_length2 + min_lon + grid_length2 / 2
    lat = loc_cell[1] * grid_length2 + min_lat + grid_length2 / 2
    return lon, lat


def build_home(train, history_order_dict, grid_length, min_lon, min_lat):
    grid_length2 = dist2angle(grid_length)

    st_grid = defaultdict(int)

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        if tid not in train['trajectories']:
            continue

        traj = train['trajectories'][tid]

        lat = traj.start_point()[1]
        lon = traj.start_point()[0]

        i = int(np.floor((lon - min_lon) / grid_length2))
        j = int(np.floor((lat - min_lat) / grid_length2))

        st_grid[(i, j)] += 1

    home_model = {
        'st_grid': st_grid,
    }

    return home_model


def generate_home_agenda(home_model, grid_length, min_lon, min_lat,
                         time_start=0, time_offset=300, time_stop=86400):
    agenda = dict()
    time_clock = time_start
    grid_length2 = dist2angle(grid_length)

    st_grid = home_model['st_grid']

    home_cell = sorted(st_grid.items(), key=lambda x: x[1], reverse=True)[0][0]

    lon, lat = get_lonlat(home_cell, grid_length2, min_lon, min_lat)

    agenda[time_clock] = [time_clock, lat, lon, 'stop']

    while time_clock < time_stop:

        time_clock += time_offset

        if time_clock < time_stop:
            lon, lat = get_lonlat(home_cell, grid_length2, min_lon, min_lat)
            agenda[time_clock] = [time_clock, lat, lon, 'stop']

    return agenda


def main():

    params = rome_params

    con = get_connection()
    cur = con.cursor()

    input_table = params['input_table']
    min_lat = params['min_lat']
    min_lon = params['min_lon']
    tzoffset = params['tzoffset']

    uid = 275299  # old_rome146099 # rome
    #uid = # old659447 # london
    week_limit = 42
    traintest_date = datetime.datetime.strptime('2015-05-03', '%Y-%m-%d')

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    # train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)
    train, test, test_days = train_test_partition_date(imh, history_order_dict, date=traintest_date)

    grid_length = 1000

    home_model = build_home(train, history_order_dict, grid_length, min_lon, min_lat)

    home_agenda = generate_home_agenda(home_model, grid_length, min_lon, min_lat,
                                             time_start=0, time_offset=300, time_stop=86400)

    print '------'

    for event_time in sorted(home_agenda):
        event = home_agenda[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]


if __name__ == "__main__":
    main()
