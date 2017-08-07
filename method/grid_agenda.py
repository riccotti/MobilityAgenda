import numpy as np

from collections import defaultdict

from util.util import *
from util.database_io import *

__author__ = 'Riccardo Guidotti'


def get_lonlat(loc_cell, grid_length2, min_lon, min_lat):
    lon = loc_cell[0] * grid_length2 + min_lon + grid_length2 / 2
    lat = loc_cell[1] * grid_length2 + min_lat + grid_length2 / 2
    return lon, lat


def get_next_st_cell(cur_cell, cur_time, st_grid_ft, random_choice=False):
    index = list(cur_cell)
    index.append(cur_time)
    index = tuple(index)

    if len(st_grid_ft[index]) == 0:
        best_candidate = None
        min_dist = float('inf')
        for index2 in st_grid_ft:
            if index != index2:
                dist = 0
                for i in range(len(index)):
                    dist += index[0]-index2[0]
                if min_dist > dist:
                    min_dist = dist
                    best_candidate = index2
        index = best_candidate

    prob = list()
    cells = list()
    for k in st_grid_ft[index]:
        cells.append(k)
        prob.append(1.0 * st_grid_ft[index][k])

    prob = np.asarray(prob)
    prob /= np.sum(prob)

    if len(st_grid_ft[index]) == 0:
        return cur_cell

    nextid = np.argmax(prob)
    if random_choice:
        nextid = np.random.choice(len(st_grid_ft[index]), 1, p=prob)[0]

    next_st_cell = cells[nextid]
    return next_st_cell


def get_next_st_cell_mov(cur_st_cell, st_grid_ft, random_choice=False):
    index = cur_st_cell

    prob = list()
    cells = list()
    for k in st_grid_ft[index]:
        cells.append(k)
        prob.append(1.0 * st_grid_ft[index][k])

    prob = np.asarray(prob)
    prob /= np.sum(prob)

    if len(st_grid_ft[index]) == 0:
        return cur_st_cell

    nextid = np.argmax(prob)

    if random_choice:
        nextid = np.random.choice(len(st_grid_ft[index]), 1, p=prob)[0]

    next_st_cell = cells[nextid]
    return next_st_cell


def build_st_grid(train, history_order_dict, grid_length, time_length, min_lon, min_lat, tzoffset):
    grid_length2 = dist2angle(grid_length)
    tzoffset = int(tzoffset * 3600 / time_length)

    st_grid = defaultdict(int)
    st_grid_ft = defaultdict(lambda: defaultdict(int))
    s_grid_timespent = defaultdict(list)
    s_grid_visits = defaultdict(int)
    s_grid_leavetime = defaultdict(list)
    st_grid_loc = dict()
    st_grid_mov = dict()

    lon_to0 = lat_to0 = time_to0 = None

    for tid in sorted(history_order_dict, key=history_order_dict.get):
        if tid not in train['trajectories']:
            continue
        traj = train['trajectories'][tid]
        i0 = j0 = h0 = None

        lon_from = float(traj.start_point()[0])
        lat_from = float(traj.start_point()[1])
        time_from = int(traj.start_point()[2])

        lon_to = float(traj.end_point()[0])
        lat_to = float(traj.end_point()[1])
        time_to = int(traj.end_point()[2])

        if lon_to0 is not None:
            i = int(np.floor((lon_to0 - min_lon) / grid_length2))
            j = int(np.floor((lat_to0 - min_lat) / grid_length2))
            h = int(((time_to0 / 1000.0) % 86400) / time_length) + tzoffset

            s_grid_timespent[(i, j)].append(time_from - time_to0)
            s_grid_visits[(i, j)] += 1
            s_grid_leavetime[(i, j)].append((time_from / 1000.0) % 86400)
            st_grid_loc[(i, j, h)] = 0

        lon_to0 = lon_to
        lat_to0 = lat_to
        time_to0 = time_to

        i = int(np.floor((lon_from - min_lon) / grid_length2))
        j = int(np.floor((lat_from - min_lat) / grid_length2))
        h = int(((time_from / 1000.0) % 86400) / time_length) + tzoffset

        s_grid_leavetime[(i, j)].append((time_from / 1000.0) % 86400)
        st_grid_loc[(i, j, h)] = 0

        for ti, xyz in enumerate(traj.object):
            lat = xyz[1]
            lon = xyz[0]

            i = int(np.floor((lon - min_lon) / grid_length2))
            j = int(np.floor((lat - min_lat) / grid_length2))
            h = int(((xyz[2] / 1000.0) % 86400) / time_length) + tzoffset

            st_grid[(i, j, h)] += 1

            if 0 < ti < len(traj.object) - 1:
                st_grid_mov[(i, j, h)] = 0

            if i0 is not None:
                st_grid_ft[(i0, j0, h0)][(i, j, h)] += 1

            i0 = i
            j0 = j
            h0 = h

    s_grid_avgts = dict()
    for k in s_grid_timespent:
        s_grid_avgts[k] = np.median(s_grid_timespent[k]) \
            if len(s_grid_timespent[k]) > 0 else 10*60*1000

    s_grid_avglt = dict()
    for k in s_grid_leavetime:
        s_grid_avglt[k] = np.median(s_grid_leavetime[k]) \
            if len(s_grid_leavetime[k]) > 0 else 0.0

    st_grid_model = {
        'st_grid': st_grid,
        'st_grid_ft': st_grid_ft,
        's_grid_timespent': s_grid_timespent,
        's_grid_visits': s_grid_visits,
        's_grid_leavetime': s_grid_leavetime,
        'st_grid_loc': st_grid_loc,
        'st_grid_mov': st_grid_mov,
        's_grid_avgts': s_grid_avgts,
        's_grid_avglt': s_grid_avglt
    }

    return st_grid_model


def generate_st_grid_agenda(st_grid_model, grid_length, time_lenght, min_lon, min_lat, tzoffset, random_choice=False,
                            time_start=0, time_offset=300, time_stop=86400, starting_cell=None):

    agenda = dict()
    time_clock = time_start

    st_grid = st_grid_model['st_grid']
    st_grid_ft = st_grid_model['st_grid_ft']
    s_grid_visits = st_grid_model['s_grid_visits']
    st_grid_loc = st_grid_model['st_grid_loc']
    st_grid_mov = st_grid_model['st_grid_mov']
    s_grid_avglt = st_grid_model['s_grid_avglt']

    # s_grid_timespent = st_grid_model['s_grid_timespent']
    # s_grid_leavetime = st_grid_model['s_grid_leavetime']
    # s_grid_avgts = st_grid_model['s_grid_avgts']

    grid_length2 = dist2angle(grid_length)

    if starting_cell is None:

        if len(s_grid_visits) == 0:
            return None, None

        starting_cell = sorted(s_grid_visits.items(), key=lambda x: x[1], reverse=True)[0][0]

    starting_time = int(s_grid_avglt[starting_cell] / time_lenght + tzoffset)
    # print starting_cell, starting_time

    is_moving = False
    next_st_cell = None

    loc_cell = starting_cell
    loc_leavetime = starting_time
    leaving_time = loc_leavetime * time_lenght
    moving_time = None

    lon, lat = get_lonlat(loc_cell, grid_length2, min_lon, min_lat)
    agenda[time_clock] = [time_clock, lat, lon, 'stop']

    while time_clock < time_stop:

        time_clock += time_offset

        if is_moving is False and time_clock - time_offset <= leaving_time <= time_clock:
            is_moving = True
            next_st_cell = get_next_st_cell(loc_cell, loc_leavetime, st_grid_ft, random_choice=random_choice)
            if len(next_st_cell) < 3:
                next_st_cell = tuple([loc_cell[0], loc_cell[1], loc_leavetime])
            moving_time = next_st_cell[2] * time_lenght
            # print 'Leaving location %s at %s' % (loc_cell, loc_leavetime), datetime.timedelta(seconds=time_clock), is_moving

        if is_moving:
            if next_st_cell not in st_grid_loc and time_clock - time_lenght <= moving_time <= time_clock:
                is_moving = True
                # next_st_cell0 = next_st_cell
                next_st_cell = get_next_st_cell_mov(next_st_cell, st_grid_ft, random_choice=random_choice)
                moving_time = next_st_cell[2] * time_lenght
                # print 'Moving from %s to %s' % (next_st_cell0, next_st_cell), datetime.timedelta(seconds=time_clock), is_moving

            if next_st_cell not in st_grid_mov and time_clock - time_lenght <= moving_time <= time_clock:
                is_moving = False
                loc_cell = tuple(list(next_st_cell)[:2])
                default_leavetime = (time_clock - tzoffset) * time_lenght + time_lenght
                loc_leavetime = int(s_grid_avglt.get(loc_cell, default_leavetime) / time_lenght + tzoffset)
                leaving_time = loc_leavetime * time_lenght
                # print 'Arriving in %s' % str(next_st_cell), datetime.timedelta(seconds=time_clock), is_moving

            if next_st_cell in st_grid_loc and next_st_cell in st_grid_mov and \
                                            time_clock - time_lenght <= moving_time <= time_clock:
                p_loc = float(st_grid[next_st_cell])
                p_mov = float(np.sum(st_grid_ft[next_st_cell].values()))
                prob = [p_loc - p_mov, p_mov]
                prob = np.asarray(prob)
                prob /= p_loc
                loc_mov = np.random.choice(2, 1, p=prob)[0]
                if loc_mov == 0:
                    is_moving = False
                    loc_cell = tuple(list(next_st_cell)[:2])
                    default_leavetime = (time_clock - tzoffset) * time_lenght + time_lenght
                    loc_leavetime = int(s_grid_avglt.get(loc_cell, default_leavetime) / time_lenght + tzoffset)
                    leaving_time = loc_leavetime * time_lenght
                    # print 'Arriving in %s' % str(next_st_cell), datetime.timedelta(seconds=time_clock), is_moving
                else:
                    is_moving = True
                    # next_st_cell0 = next_st_cell
                    next_st_cell = get_next_st_cell_mov(next_st_cell, st_grid_ft, random_choice=random_choice)
                    moving_time = next_st_cell[2] * time_lenght
                    # print 'Moving from %s to %s' % (next_st_cell0, next_st_cell), datetime.timedelta(seconds=time_clock), is_moving

        if time_clock < time_stop:
            if is_moving:
                lon, lat = get_lonlat(next_st_cell, grid_length2, min_lon, min_lat)
                agenda[time_clock] = [time_clock, lat, lon, 'move']
            else:
                lon, lat = get_lonlat(loc_cell, grid_length2, min_lon, min_lat)
                agenda[time_clock] = [time_clock, lat, lon, 'stop']

    ending_cell = loc_cell

    return agenda, ending_cell


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

    grid_model = build_st_grid(train, history_order_dict, grid_length, time_lenght, min_lon, min_lat, tzoffset)

    grid_agenda = generate_st_grid_agenda(grid_model, grid_length, time_lenght, min_lon, min_lat, tzoffset,
                                          time_start=0, time_offset=300, time_stop=86400)

    print '------'

    for event_time in sorted(grid_agenda):
        event = grid_agenda[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]

if __name__ == "__main__":
    main()
