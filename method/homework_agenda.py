import numpy as np

from collections import defaultdict

from util.util import *
from util.database_io import *


__author__ = 'Riccardo Guidotti'


def normalize(x):

    max_val = np.sum(x)

    if max_val == 0:
        return [1.0, 0.0]

    nx = list()
    for x0 in x:
        nx.append(1.0 * x0 / max_val)

    return nx


def get_lonlat(loc_cell, grid_length2, min_lon, min_lat):
    lon = loc_cell[0] * grid_length2 + min_lon + grid_length2 / 2
    lat = loc_cell[1] * grid_length2 + min_lat + grid_length2 / 2
    return lon, lat


def build_homework(train, history_order_dict, grid_length, time_length, min_lon, min_lat, tzoffset):
    grid_length2 = dist2angle(grid_length)
    tzoffset = int(tzoffset * 3600 / time_length)

    s_grid = defaultdict(int)
    st_grid = defaultdict(lambda: defaultdict(int))

    lon_to0 = lat_to0 = time_to0 = None

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        if tid not in train['trajectories']:
            continue

        traj = train['trajectories'][tid]

        lat = traj.start_point()[1]
        lon = traj.start_point()[0]
        i = int(np.floor((lon - min_lon) / grid_length2))
        j = int(np.floor((lat - min_lat) / grid_length2))
        s_grid[(i, j)] += 1

        time_from = int(traj.start_point()[2])
        lon_to = float(traj.end_point()[0])
        lat_to = float(traj.end_point()[1])
        time_to = int(traj.end_point()[2])

        if lon_to0 is not None:
            i = int(np.floor((lon_to0 - min_lon) / grid_length2))
            j = int(np.floor((lat_to0 - min_lat) / grid_length2))

            # h0 = datetime.datetime.fromtimestamp(time_to0 / 1000.0)
            # h1 = datetime.datetime.fromtimestamp(time_from / 1000.0)
            #
            # for dt in rrule(freq=HOURLY, dtstart=h0, until=h1):
            #     h = (dt.time().hour + tzoffset) % 24
            #     st_grid[(i, j)][h] += 1

            ts1 = datetime.datetime.fromtimestamp(time_to0 / 1000 + tzoffset * 3600)
            ts2 = datetime.datetime.fromtimestamp(time_from / 1000 + tzoffset * 3600)

            at = ts1.replace(second=0, microsecond=0)
            lt = ts2.replace(second=0, microsecond=0)

            midnight_at = at.replace(hour=0, minute=0)
            midnight_lt = lt.replace(hour=0, minute=0)

            at_sec = int((at - midnight_at).total_seconds())
            lt_sec = int((lt - midnight_lt).total_seconds())

            # print at_sec, lt_sec

            at_sec = int(np.round(at_sec / time_length) * time_length)
            lt_sec = int(np.round(lt_sec / time_length) * time_length)

            # print at_sec, lt_sec

            # for h in range(at_sec, lt_sec + time_length, time_length):
            #     st_grid[(i, j)][h] += 1

            if at_sec <= lt_sec:
                for h in range(at_sec, lt_sec + time_length, time_length):
                    st_grid[(i, j)][h] += 1
            elif at_sec > lt_sec:
                for h in range(0, lt_sec + time_length, time_length):
                    st_grid[(i, j)][h] += 1
                for h in range(at_sec, 86400, time_length):
                    st_grid[(i, j)][h] += 1

        lon_to0 = lon_to
        lat_to0 = lat_to
        time_to0 = time_to

    homework_model = {
        's_grid': s_grid,
        'st_grid': st_grid,
    }

    return homework_model


def generate_homework_agenda(home_model, grid_length, time_length, min_lon, min_lat,
                             time_start=0, time_offset=300, time_stop=86400):
    agenda = dict()
    time_clock = time_start
    grid_length2 = dist2angle(grid_length)

    s_grid = home_model['s_grid']
    st_grid = home_model['st_grid']

    if len(s_grid) == 0:
        return None

    home_cell = sorted(s_grid.items(), key=lambda x: x[1], reverse=True)[0][0]
    if len(s_grid) > 1:
        work_cell = sorted(s_grid.items(), key=lambda x: x[1], reverse=True)[1][0]
    else:
        work_cell = home_cell

    home_time = st_grid[home_cell]
    work_time = st_grid[work_cell]

    last_place = 'home'

    # print s_grid[home_cell], s_grid[work_cell]
    # print home_cell, home_time
    # print work_cell, work_time

    lon, lat = get_lonlat(home_cell, grid_length2, min_lon, min_lat)
    agenda[time_clock] = [time_clock, lat, lon, 'stop']

    while time_clock < time_stop:

        time_clock += time_offset

        if time_clock < time_stop:

            h = time_clock  # / time_length

            if home_time[h] + work_time[h] > 0:
                hw = np.random.choice(['home', 'work'], p=normalize([home_time[h], work_time[h]]))
            else:
                hw = last_place

            if hw == 'home':
                lon, lat = get_lonlat(home_cell, grid_length2, min_lon, min_lat)
            if hw == 'work':
                lon, lat = get_lonlat(work_cell, grid_length2, min_lon, min_lat)

            last_place = hw

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

    uid = 146099 # rome
    #uid = 659447 # london
    week_limit = 42

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)

    grid_length = 1000
    time_lenght = 3600

    homework_model = build_homework(train, history_order_dict, grid_length, time_lenght,
                                    min_lon, min_lat, tzoffset)

    homework_agenda = generate_homework_agenda(homework_model, grid_length, time_lenght, min_lon, min_lat,
                                               time_start=0, time_offset=300, time_stop=86400)

    # print '------'
    #
    # for event_time in sorted(homework_agenda):
    #     event = homework_agenda[event_time]
    #     print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]


if __name__ == "__main__":
    main()
