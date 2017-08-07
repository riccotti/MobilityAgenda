import copy

import matplotlib.pyplot as plt

from util.util import *
from util.database_io import *

from myagenda.tosca import *
from myagenda.bisecting_kmeans import *


__author__ = 'Riccardo Guidotti'


def get_points_trajfromto(imh):

    trajectories = imh['trajectories']

    points = dict()
    traj_from_to = dict()

    for tid, traj in trajectories.iteritems():

        lon_from = float(traj.start_point()[0])
        lat_from = float(traj.start_point()[1])
        time_from = int(traj.start_point()[2])

        lon_to = float(traj.end_point()[0])
        lat_to = float(traj.end_point()[1])
        time_to = int(traj.end_point()[2])

        pid_start_point = len(points)
        points[pid_start_point] = [lon_from, lat_from, time_from, 'f', tid]

        pid_end_point = len(points)
        points[pid_end_point] = [lon_to, lat_to, time_to, 't', tid]

        traj_from_to[tid] = [pid_start_point, pid_end_point]

    return points, traj_from_to


def radius_of_gyration(points, centre_of_mass, dist):
    rog = 0
    for p in points:
        rog += dist(p, centre_of_mass)
    rog = 1.0*rog/len(points)
    return rog


def entropy(x, classes=None):
    if len(x) == 1:
        return 0.0
    val_entropy = 0
    n = np.sum(x)
    for freq in x:
        if freq == 0:
            continue
        p = 1.0 * freq / n
        val_entropy -= p * np.log2(p)
    if classes is not None and classes:
        val_entropy /= np.log2(classes)
    return val_entropy


def normalize_dict(x):

    max_val = np.max(x.values())

    nx = dict()
    for k in x:
        nx[k] = 1.0 * x[k] / max_val

    return nx

# def get_convex_hull(points):
#     p_list = list()
#
#     for p in points:
#         p_list.append((p[0], p[1]))
#
#     if len(p_list) < 3:
#         geos = GeoSeries([Point(p_list[0])])
#     else:
#         geos = GeoSeries([Polygon(p_list)])
#
#     gdf = gpd.GeoDataFrame({'geometry': geos.convex_hull})
#
#     convex_hull_shape = json.loads(gdf.to_json())['features'][0]['geometry']
#
#     return convex_hull_shape


def locations_detection(points, min_dist=50.0, nrun=5):

    # npoints = len(points)

    spatial_points = list()
    for p in points.values():
        if p[0] in [np.nan, np.inf] or p[1] in [np.nan, np.inf]:
            continue
        spatial_points.append(p[0:2])

    if len(spatial_points) == 0:
        return None

    centers_min, centers_max = get_min_max(spatial_points)

    cluster_res = dict()
    cuts = dict()
    for runid in range(0, nrun):
        try:
            tosca = Tosca(kmin=centers_min, kmax=centers_max, xmeans_df=spherical_distances,
                          singlelinkage_df=spherical_distance, is_outlier=thompson_test,
                          min_dist=min_dist, verbose=False)
            tosca.fit(np.asarray(spatial_points))
            cluster_res[tosca.k_] = tosca.cluster_centers_
            cuts[tosca.k_] = tosca.cut_dist_
        except ValueError:
            pass
        # # cut_dist = tosca.cut_dist_
        # # print cut_dist
        # # nlocations_clusters = tosca.k_

        # bkmeans = BisectiveKmeans(250.0, distances=spherical_distances, distance=spherical_distance)
        # bkmeans.fit(np.asarray(spatial_points))
        # cluster_res[bkmeans.k_] = bkmeans.cluster_centers_

    if len(cluster_res) == 0:
        return None

    index = np.min(cluster_res.keys())
    centers = cluster_res[index]
    loc_tosca_cut = cuts[index]

    # calculate distances between points and medoids
    distances = spherical_distances(spatial_points, centers)

    # calculates labels according to minimum distance
    labels = np.argmin(distances, axis=1)

    # build clusters according to labels and assign point to point identifier
    location_points = defaultdict(list)
    location_prototype = dict()
    for pid, lid in enumerate(labels):
        location_points[lid].append(pid)
        location_prototype[lid] = list(centers[lid])

    pid_lid = dict()
    location_support = dict()
    for lid in location_points:
        location_support[lid] = len(location_points[lid])
        for pid in location_points[lid]:
            pid_lid[pid] = lid

    # statistical information for users analysis
    cm = np.mean(spatial_points, axis=0)
    rg = radius_of_gyration(spatial_points, cm, spherical_distance)
    en = entropy(location_support.values(), classes=len(location_support))

    res = {
        'location_points': location_points,
        'location_prototype': location_prototype,
        'pid_lid': pid_lid,
        'rg': rg,
        'entropy': en,
        'loc_tosca_cut': loc_tosca_cut,
    }

    return res


def movements_detection(pid_lid, traj_from_to, imh):

    traj_from_to_loc = dict()
    loc_from_to_traj = defaultdict(list)
    loc_nextlocs = defaultdict(lambda: defaultdict(int))
    for tid, from_to in traj_from_to.iteritems():
        loc_from = pid_lid[from_to[0]]
        loc_to = pid_lid[from_to[1]]
        traj_from_to_loc[tid] = [loc_from, loc_to]
        loc_from_to_traj[(loc_from, loc_to)].append(tid)
        loc_nextlocs[loc_from][loc_to] += 1

    movement_traj = dict()
    lft_mid = dict()
    for mid, lft in enumerate(loc_from_to_traj):
        movement_traj[mid] = [lft, loc_from_to_traj[lft]]
        lft_mid[lft] = mid

    trajectories = imh['trajectories']
    movement_prototype = dict()

    for mid in movement_traj:
        traj_in_movement = movement_traj[mid][1]

        if len(traj_in_movement) > 2:
            prototype = None
            min_dist = float('inf')
            for tid1 in traj_in_movement:
                tot_dist = 0.0
                traj1 = trajectories[tid1]
                for tid2 in traj_in_movement:
                    traj2 = trajectories[tid2]
                    dist = trajectory_distance(traj1, traj2)
                    tot_dist += dist
                if tot_dist < min_dist:
                    min_dist = tot_dist
                    prototype = traj1
            movement_prototype[mid] = prototype
        else:
            movement_prototype[mid] = trajectories[traj_in_movement[0]]

    res = {
        'movement_traj': movement_traj,
        'movement_prototype': movement_prototype,
        'loc_nextlocs': loc_nextlocs,
        'traj_from_to_loc': traj_from_to_loc,
        'lft_mid': lft_mid,
    }

    return res


def get_location_features(points_in_loc, traj_from_to_loc, location_prototype, regular_locs, imh, tzoffset):

    # print points_in_loc
    trajectories = imh['trajectories']
    sorted_points = sorted(points_in_loc, key=lambda x: points_in_loc[x][2])

    staytime_dist = defaultdict(int)
    nextloc_count = defaultdict(int)
    nextloc_dist = defaultdict(lambda: defaultdict(int))

    for i in range(0, len(sorted_points)-1):
        pid1 = sorted_points[i]
        pid2 = sorted_points[i+1]

        arriving_leaving = points_in_loc[pid1][3]

        if arriving_leaving == 't':
            ts1 = datetime.datetime.fromtimestamp(points_in_loc[pid1][2] / 1000 + tzoffset * 3600)
            ts2 = datetime.datetime.fromtimestamp(points_in_loc[pid2][2] / 1000 + tzoffset * 3600)

            # print pid1, ts1, points_in_loc[pid1][3], traj_from_to_loc[points_in_loc[pid1][4]]
            # print pid2, ts2, points_in_loc[pid2][3], traj_from_to_loc[points_in_loc[pid2][4]]

            at = ts1.replace(second=0, microsecond=0)
            lt = ts2.replace(second=0, microsecond=0)

            midnight_at = at.replace(hour=0, minute=0)
            midnight_lt = lt.replace(hour=0, minute=0)

            at_sec = int((at - midnight_at).total_seconds())
            lt_sec = int((lt - midnight_lt).total_seconds())

            # print at, at_sec
            # print lt, lt_sec
            # print ''
            # datetime.datetime.combine(datetime.datetime(1, 1, 1), a)

            if at_sec <= lt_sec:
                for minute in range(at_sec, lt_sec + 60, 60):
                    dt_minute = datetime.time(hour=minute/3600, minute=(minute % 3600)/60)
                    staytime_dist[dt_minute] += 1
            elif at_sec > lt_sec:
                for minute in range(0, lt_sec + 60, 60):
                    dt_minute = datetime.time(hour=minute/3600, minute=(minute % 3600)/60)
                    staytime_dist[dt_minute] += 1
                for minute in range(at_sec, 86400, 60):
                    dt_minute = datetime.time(hour=minute / 3600, minute=(minute % 3600) / 60)
                    staytime_dist[dt_minute] += 1

            # for minute in range(0, 86400 + 60, 60):
            #     if at_sec <= minute <= lt_sec:
            #         dt_minute = datetime.time(hour=minute/3600, minute=(minute % 3600)/60)
            #         staytime_dist[dt_minute] += 1

            # for minute in range(at_sec, lt_sec + 60, 60):
            #     dt_minute = datetime.time(hour=minute/3600, minute=(minute % 3600)/60)
            #     staytime_dist[dt_minute] += 1
            # print ''

        if arriving_leaving == 'f':
            tid = points_in_loc[pid1][4]
            # cur_loc = traj_from_to_loc[tid][0]

            next_loc = traj_from_to_loc[tid][1]

            if next_loc not in regular_locs:
                continue

            nextloc_count[next_loc] += 1

            ts1 = datetime.datetime.fromtimestamp(trajectories[tid].start_point()[2] / 1000 + tzoffset * 3600)
            ts2 = datetime.datetime.fromtimestamp(trajectories[tid].end_point()[2] / 1000 + tzoffset * 3600)

            at = ts1.replace(second=0, microsecond=0)
            lt = ts2.replace(second=0, microsecond=0)

            midnight_at = at.replace(hour=0, minute=0)
            midnight_lt = lt.replace(hour=0, minute=0)

            at_sec = int((at - midnight_at).total_seconds())
            lt_sec = int((lt - midnight_lt).total_seconds())

            for minute in range(at_sec, lt_sec + 60, 60):
                dt_minute = datetime.time(hour=minute/3600, minute=(minute % 3600)/60)
                nextloc_dist[next_loc][dt_minute] += 1
            # # datetime.datetime.combine(datetime.datetime(1, 1, 1), a)

    # x_ticks = list()
    # x_ticks_idx = list()
    # y_list = list()
    # for i, ts in enumerate(sorted(staytime_dist)):
    #     # print ts, loc_dist[ts]
    #     if i % 50 == 0:
    #         x_ticks.append(str(ts))
    #         x_ticks_idx.append(i)
    #     # y_list.append(sigmoid(staytime_dist[ts]))
    #     y_list.append(staytime_dist[ts])
    #
    # plt.plot(range(0, len(y_list)), y_list, linewidth=1)
    # # plt.plot(range(0, len(y_list)), smooth(y_list, 20), linewidth=1)
    # # plt.plot(range(0, len(y_list)), savgol_filter(y_list, 9, 5), linewidth=1)
    # plt.xticks(x_ticks_idx, x_ticks, rotation='vertical')
    # plt.show()
    #
    # x_ticks = dict()
    # for next_loc in nextloc_dist:
    #     y_list = list()
    #     for ts in sorted(nextloc_dist[next_loc]):
    #         x_ticks[str(ts)] = 0
    #         y_list.append(nextloc_dist[next_loc][ts])
    #
    #     plt.plot(range(0, len(y_list)), y_list, linewidth=1)
    #
    # x_ticks_idx = list()
    # x_ticks_label = list()
    # for i, t in enumerate(sorted(x_ticks)):
    #     if i % max(10, len(x_ticks)/10) == 0:
    #         x_ticks_idx.append(i)
    #         x_ticks_label.append(t)
    # plt.xticks(x_ticks_idx, x_ticks_label, rotation='vertical')
    # plt.show()

    spatial_points = list()
    for p in points_in_loc.values():
        spatial_points.append(p[0:2])

    loc_rg = radius_of_gyration(spatial_points, location_prototype, spherical_distance)
    loc_entropy = entropy(nextloc_count.values(), classes=len(nextloc_count))

    res = {
        'staytime_dist': staytime_dist,
        'nextloc_dist': nextloc_dist,
        'nextloc_count': nextloc_count,
        'loc_rg': loc_rg,
        'loc_entropy': loc_entropy,
    }

    return res


def get_locations_features(points, traj_from_to_loc, location_points, location_prototype, regular_locs, imh, tzoffset):
    res = dict()

    for lid in location_points:

        if lid not in regular_locs:
            continue

        points_in_loc = dict()
        for pid in location_points[lid]:
            points_in_loc[pid] = points[pid]

        lf_res = get_location_features(points_in_loc, traj_from_to_loc, location_prototype[lid],
                                       regular_locs, imh, tzoffset)

        lf_res['loc_support'] = len(location_points[lid])

        res[lid] = lf_res

    staytime_tot_dist = defaultdict(list)
    for lid in res:
        staytime_dist = res[lid]['staytime_dist']

        for ts in staytime_dist:
            staytime_tot_dist[ts].append(staytime_dist[ts])

    staytime_totals = dict()
    for ts in staytime_tot_dist:
        staytime_totals[ts] = np.sum(staytime_tot_dist[ts])

    for lid in res:
        staytime_dist = res[lid]['staytime_dist']
        staytime_ndist = dict()
        for ts in staytime_dist:
            staytime_ndist[ts] = 1.0 * staytime_dist[ts] / staytime_totals[ts]
        res[lid]['staytime_ndist'] = staytime_ndist

    return res


def interquartile_filter(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    y = list()
    for x0 in x:
        if q1 - 1.5 * iqr <= x0 <= q3 + 1.5 * iqr:
            y.append(x0)
    return y


def get_movements_features(movement_traj, imh):

    trajectories = imh['trajectories']

    res = dict()
    for mid in movement_traj:
        traj_in_movement = movement_traj[mid][1]
        movement_support = len(traj_in_movement)
        movement_lengths = list()
        movement_durations = list()
        for tid in traj_in_movement:
            movement_lengths.append(trajectories[tid].length())
            movement_durations.append(trajectories[tid].duration()/1000)

        movement_lengths = interquartile_filter(movement_lengths)
        movement_durations = interquartile_filter(movement_durations)

        res[mid] = {
            'mov_support': movement_support,
            'typical_mov_length': np.median(movement_lengths),
            'avg_mov_length': np.mean(movement_lengths),
            'std_mov_length': np.std(movement_lengths),
            'typical_mov_duration': datetime.timedelta(seconds=np.median(movement_durations)),
            'avg_mov_duration': datetime.timedelta(seconds=np.mean(movement_durations)),
            'std_mov_duration': datetime.timedelta(seconds=np.std(movement_durations)),
        }

    return res


def get_movements_stats(movement_traj, regular_locs, imh):
    trajectories = imh['trajectories']

    movement_lengths = list()
    movement_durations = list()
    reg_movement_lengths = list()
    reg_movement_durations = list()
    reg_movs = dict()
    n_reg_traj = 0

    for mid in movement_traj:
        lft = movement_traj[mid][0]
        traj_in_movement = movement_traj[mid][1]
        for tid in traj_in_movement:
            movement_lengths.append(trajectories[tid].length())
            movement_durations.append(trajectories[tid].duration() / 1000)
            if lft[0] in regular_locs and lft[1] in regular_locs:
                reg_movs[mid] = 0
                reg_movement_lengths.append(trajectories[tid].length())
                reg_movement_durations.append(trajectories[tid].duration() / 1000)
                n_reg_traj += 1

    movement_lengths = interquartile_filter(movement_lengths)
    movement_durations = interquartile_filter(movement_durations)

    if len(reg_movement_lengths) > 0:
        reg_movement_lengths = interquartile_filter(reg_movement_lengths)
        reg_movement_durations = interquartile_filter(reg_movement_durations)

    avg_mov_duration = datetime.timedelta(seconds=np.mean(movement_durations))
    std_mov_duration = datetime.timedelta(seconds=np.std(movement_durations))

    if len(reg_movement_lengths) > 0:
        avg_reg_mov_duration = datetime.timedelta(seconds=np.mean(reg_movement_durations))
        std_reg_mov_duration = datetime.timedelta(seconds=np.std(reg_movement_durations))
    else:
        avg_reg_mov_duration = avg_mov_duration
        std_reg_mov_duration = std_mov_duration

    res = {
        'n_reg_movs': len(reg_movs),
        'avg_mov_length': np.mean(movement_lengths),
        'std_mov_length': np.std(movement_lengths),
        'avg_mov_duration': avg_mov_duration,
        'std_mov_duration': std_mov_duration,
        'avg_reg_mov_length': np.mean(reg_movement_lengths),
        'std_reg_mov_length': np.std(reg_movement_lengths),
        'avg_reg_mov_duration': avg_reg_mov_duration,
        'std_reg_mov_duration': std_reg_mov_duration,
        'n_reg_traj': n_reg_traj,
    }

    return res


def closest_point_on_segment_minsup(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)

    if u < 0.00001:
        closest_point = a
    elif u > 1:
        closest_point = b
    else:
        cp_x = sx1 + u * x_delta
        cp_y = sy1 + u * y_delta
        closest_point = [cp_x, cp_y]

    return closest_point


def get_minimum_support(locations_support):
    x = []
    y = []

    sorted_support = sorted(locations_support)

    for i, s in enumerate(sorted_support):
        x.append(1.0 * i)
        y.append(1.0 * s)

    # plt.plot(x, y)
    # plt.plot([0, len(x) - 1], [y[0], y[len(y)-1]])

    max_d = -float('infinity')
    index = 0

    a = [x[0], y[0]]
    b = [x[len(x)-1], y[len(y)-1]]

    for i in range(0, len(x)):
        p = [x[i], y[i]]
        c = closest_point_on_segment_minsup(a, b, p)
        d = math.sqrt((c[0]-x[i])**2 + (c[1]-y[i])**2)

        if d > max_d:
            max_d = d
            index = i

    # c = closest_point_on_segment_minsup(a, b, [x[index], y[index]] )
    # plt.plot([x[index], c[0]], [y[index], c[1]])
    # plt.show()

    return sorted_support[index]


def detect_regular_locations(location_points, loc_nextlocs):

    loc_support = dict()

    for lid in location_points:
        loc_support[lid] = len(location_points[lid])

    # print sorted(loc_support.values(), reverse=True)

    loc_min_sup = get_minimum_support(loc_support.values())
    # print '>>>>>>', loc_min_sup

    regular_locs = dict()
    for lid in loc_support:
        if loc_support[lid] >= loc_min_sup:
            regular_locs[lid] = loc_nextlocs[lid]

    is_dag = False
    while not is_dag and not len(regular_locs) <= 2:
        # print is_dag, len(regular_locs)
        is_dag = True
        for lid in regular_locs:
            has_an_out_mov = False
            for lid_out in regular_locs[lid]:
                if lid_out in regular_locs and lid_out != lid:
                    has_an_out_mov = True
                    break
            if not has_an_out_mov:
                del regular_locs[lid]
                is_dag = False
                break

    return regular_locs, loc_min_sup, is_dag


def caclulate_regular_rgen(regular_locs, loc_res, points):

    spatial_points = list()
    rloc_support = dict()
    for rlid in regular_locs:
        rloc_support[rlid] = len(loc_res['location_points'][rlid])
        for pid in loc_res['location_points'][rlid]:
            p = points[pid]
            spatial_points.append(p[0:2])

    cm = np.mean(spatial_points, axis=0)
    rrg = radius_of_gyration(spatial_points, cm, spherical_distance)
    ren = entropy(rloc_support.values(), classes=len(rloc_support))

    return rrg, ren


def build_myagenda(imh, tzoffset, reg_loc=True):

    n_traj = len(imh['trajectories'])

    points, traj_from_to = get_points_trajfromto(imh)
    loc_res = locations_detection(points)

    if loc_res is None:
        return None

    mov_res = movements_detection(loc_res['pid_lid'], traj_from_to, imh)

    n_locs = len(loc_res['location_points'])
    n_movs = len(mov_res['movement_traj'])

    rrg = ren = loc_min_sup = None
    regular_locs = loc_res['location_points']
    n_reg_locs = len(regular_locs)

    if reg_loc:
        regular_locs, loc_min_sup, is_dag = detect_regular_locations(
            loc_res['location_points'], mov_res['loc_nextlocs'])
        n_reg_locs = len(regular_locs)
        rrg, ren = caclulate_regular_rgen(regular_locs, loc_res, points)

        # print len(regular_locs), loc_min_sup, is_dag
        # for lid in loc_res['location_points']:
        #     if len(loc_res['location_points'][lid]) >= 5:
        #         regular_locs[lid] = 0

    lf_res = get_locations_features(points, mov_res['traj_from_to_loc'], loc_res['location_points'],
                                    loc_res['location_prototype'], regular_locs, imh, tzoffset)

    mf_res = get_movements_features(mov_res['movement_traj'], imh)
    ms_res = get_movements_stats(mov_res['movement_traj'], regular_locs, imh)

    myagenda = {
        'loc_points': loc_res['location_points'],
        'loc_prototype': loc_res['location_prototype'],
        'loc_features': lf_res,
        'loc_nextlocs': mov_res['loc_nextlocs'],
        'mov_traj': mov_res['movement_traj'],
        'mov_prototype': mov_res['movement_prototype'],
        'mov_features': mf_res,
        'pid_lid': loc_res['pid_lid'],
        'traj_from_to_loc': mov_res['traj_from_to_loc'],
        'lft_mid': mov_res['lft_mid'],

        'n_traj': n_traj,
        'n_reg_traj': ms_res['n_reg_traj'],
        'n_locs': n_locs,
        'n_reg_locs': n_reg_locs,
        'n_movs': n_movs,
        'n_reg_movs': ms_res['n_reg_movs'],
        'rg': loc_res['rg'],
        'rrg': rrg,
        'entropy': loc_res['entropy'],
        'rentropy': ren,
        'avg_mov_length': ms_res['avg_mov_length'],
        'std_mov_length': ms_res['std_mov_length'],
        'avg_mov_duration': ms_res['avg_mov_duration'],
        'std_mov_duration': ms_res['std_mov_duration'],
        'avg_reg_mov_length': ms_res['avg_reg_mov_length'],
        'std_reg_mov_length': ms_res['std_reg_mov_length'],
        'avg_reg_mov_duration': ms_res['avg_reg_mov_duration'],
        'std_reg_mov_duration': ms_res['std_reg_mov_duration'],
        'loc_tosca_cut': loc_res['loc_tosca_cut'],
        'loc_sup_cut': loc_min_sup,
    }

    return myagenda


def get_lonlat(location):
    lon = location[0]
    lat = location[1]
    return lon, lat


# def time2sec(ts):
#     ts_datetime = datetime.datetime.combine(datetime.datetime(1, 1, 1), ts)
#     midnight = ts_datetime.replace(hour=0, minute=0)
#     sec = int((ts_datetime - midnight).total_seconds())
#     return sec
#
# def sec2time(sec):
#     sec = int((ts_datetime - midnight).total_seconds())
#
#     ts_datetime = datetime.datetime.combine(datetime.datetime(1, 1, 1), ts)
#     midnight = ts_datetime.replace(hour=0, minute=0)
#     return ts


def sigmoid(x, x0=0.5, k=10.0, L=1.0):
    return L / (1 + np.exp(-k * (x - x0)))


def get_next_location(cur_lid, time_clock, time_offset, time_stop, cur_loc_features, random_choice=False):
    staytime_ndist = cur_loc_features['staytime_ndist']
    # staytime_ndist = normalize_dict(staytime_dist)

    stay_probs = list()
    time_clock_from = abs((time_clock - np.floor(time_offset/(60*2.0))*60) % time_stop)
    # time_clock_to = abs((time_clock + np.ceil(time_offset/(60*2.0))*60) % time_stop)

    nclocks = time_offset / 60
    for cloc_idx in range(0, nclocks):
        sec = int(abs((time_clock_from + (60 * cloc_idx)) % time_stop))
        hour = sec / 3600
        minute = (sec % 3600) / 60
        ts = datetime.time(hour, minute)
        # stay_probs.append(staytime_ndist[ts])
        stay_probs.append(staytime_ndist.get(ts, np.median(staytime_ndist.values())))

    p_stay = sigmoid(1.0 * np.median(stay_probs))
    stay_leave = ['stay', 'leave'][np.argmax([p_stay, 1.0 - p_stay])]

    if random_choice:
        stay_leave = np.random.choice(['stay', 'leave'], p=[p_stay, 1.0 - p_stay])

    if stay_leave == 'stay':
        return cur_lid

    if stay_leave == 'leave':
        nextloc_dist = cur_loc_features['nextloc_dist']

        next_lid_prob = dict()
        for next_lid in nextloc_dist:

            leave_from_next_lid_probs = list()
            for cloc_idx in range(0, nclocks):
                sec = int(abs((time_clock_from + (60 * cloc_idx)) % time_stop))
                ts = datetime.time(sec / 3600, (sec % 3600) / 60)
                leave_from_next_lid_probs.append(nextloc_dist[next_lid][ts])

            next_lid_prob[next_lid] = 1.0 * np.median(leave_from_next_lid_probs)

        # versione sigmoide
        next_lid_nprob = dict()
        tot_leave_count = np.sum(next_lid_prob.values())
        for k, v in next_lid_prob.iteritems():
            if v > 0.0:
                next_lid_nprob[k] = sigmoid(1.0 * v / tot_leave_count)
        next_lid_nprob = next_lid_prob

        if len(next_lid_nprob) == 0:
            # non ho orari di uscita ora quindi mi fermo qui
            return cur_lid

        tot_prob = np.sum(next_lid_nprob.values())
        for k, v in next_lid_nprob.iteritems():
            next_lid_nprob[k] = 1.0 * v / tot_prob

        probs = [next_lid_nprob[lid] for lid in sorted(next_lid_nprob)]

        next_lid = sorted(next_lid_nprob)[np.argmax(probs)]

        if random_choice:
            next_lid = np.random.choice(sorted(next_lid_nprob), p=probs)

        return next_lid

    return None


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def get_temporal_aligned_traj(traj, ta_time_start, ratio_dur):
    ta_traj = Trajectory(id='tmp', object=copy.deepcopy(traj.object), vehicle='tmp')
    ta_traj.start_point()[2] = ta_time_start

    for i in range(1, len(traj)):
        rt0 = (traj.point_n(i - 1)[2] / 1000) % 86400
        rt1 = (traj.point_n(i)[2] / 1000) % 86400
        time_diff = np.round(((rt1 - rt0) * ratio_dur))
        ta_traj.point_n(i)[2] = ta_traj.point_n(i - 1)[2] + time_diff

    return ta_traj


def get_movement_duration(cur_mid, mov_features, random_choice=False):
    dur_mu = mov_features[cur_mid]['avg_mov_duration'].total_seconds()
    dur_sigma = mov_features[cur_mid]['std_mov_duration'].total_seconds()
    if random_choice and 1.0 * dur_sigma / dur_mu < 1.0 and dur_sigma > 0.0:
        mov_duration = np.random.normal(dur_mu, dur_sigma)
    else:
        mov_duration = mov_features[cur_mid]['typical_mov_duration'].total_seconds()

    # mov_duration = mov_features[cur_mid]['typical_mov_duration'].total_seconds()

    # dist = np.random.normal(mu, sigma, 1000)
    # count, bins, ignored = plt.hist(dist, 30, normed=True)
    # plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
    # linewidth = 2, color = 'r')
    # plt.show()

    return mov_duration


def generate_myagenda(myagenda_model, random_choice=False,
                      time_start=0, time_offset=300, time_stop=86400, starting_lid=None):

    agenda = dict()
    time_clock = time_start

    loc_prototype = myagenda_model['loc_prototype']
    loc_features = myagenda_model['loc_features']
    # loc_nextlocs = myagenda_model['loc_nextlocs']
    lft_mid = myagenda_model['lft_mid']
    mov_prototype = myagenda_model['mov_prototype']
    mov_features = myagenda_model['mov_features']

    is_moving = False
    cur_mov = None
    ta_cur_mov = None
    cur_mov_idx = 1

    if starting_lid is None:
        starting_lid = sorted(loc_features.items(), key=lambda x: x[1]['loc_support'], reverse=True)[0][0]

    cur_lid = starting_lid
    lon, lat = get_lonlat(loc_prototype[cur_lid])
    agenda[time_clock] = [time_clock, lat, lon, 'stop']

    while time_clock < time_stop:

        time_clock += time_offset

        if not is_moving and time_clock < time_stop:

            next_lid = get_next_location(cur_lid, time_clock, time_offset, time_stop, loc_features[cur_lid],
                                         random_choice)
            is_moving = not next_lid == cur_lid

            if not is_moving:
                lon, lat = get_lonlat(loc_prototype[cur_lid])
                agenda[time_clock] = [time_clock, lat, lon, 'stop']

            if is_moving:

                # # per plot
                # staytime_ndist = loc_features[cur_lid]['staytime_ndist']
                # # staytime_ndist = normalize_dict(staytime_dist)
                # x_ticks = list()
                # x_ticks_idx = list()
                # y_list = list()
                # for i, ts in enumerate(sorted(staytime_ndist)):
                #     # print ts, loc_dist[ts]
                #     if i % 50 == 0:
                #         x_ticks.append(str(ts))
                #         x_ticks_idx.append(i)
                #     y_list.append(sigmoid(staytime_ndist[ts]))
                #
                # plt.plot(range(0, len(y_list)), y_list, linewidth=1)
                # plt.plot(range(0, len(y_list)), smooth(y_list, 20), linewidth=1)
                # # plt.plot(range(0, len(y_list)), savgol_filter(y_list, 9, 5), linewidth=1)
                # plt.xticks(x_ticks_idx, x_ticks, rotation='vertical')
                # plt.show()
                # # per plot

                lon, lat = get_lonlat(loc_prototype[cur_lid])
                # print 'Leaving location (%s, %s)' % (lon, lat), datetime.timedelta(seconds=time_clock), is_moving

                # print lat, lon
                cur_mid = lft_mid[(cur_lid, next_lid)]
                cur_mov = mov_prototype[lft_mid[(cur_lid, next_lid)]]
                cur_lid = next_lid

                # print cur_mid
                # print cur_mov.length(), datetime.timedelta(seconds=cur_mov.duration()/1000)
                # print mov_features[cur_mid]['typical_mov_length'], mov_features[cur_mid]['typical_mov_duration']
                # print mov_features[cur_mid]['avg_mov_length'], mov_features[cur_mid]['avg_mov_duration']
                # print mov_features[cur_mid]['std_mov_length'], mov_features[cur_mid]['std_mov_duration']
                # print ''

                mov_duration = get_movement_duration(cur_mid, mov_features)
                real_duration = cur_mov.duration() / 1000
                ratio_dur = 1.0 * mov_duration / real_duration
                # print mov_duration, real_duration, ratio_dur

                ta_time_start = time_clock - time_offset
                ta_cur_mov = get_temporal_aligned_traj(cur_mov, ta_time_start, ratio_dur)

                while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                    cur_mov_idx += 1

                # print 'cur_mov_idx', cur_mov_idx

                if cur_mov_idx < len(cur_mov):
                    mov_a = ta_cur_mov.point_n(cur_mov_idx - 1)
                    mov_b = ta_cur_mov.point_n(cur_mov_idx)
                    time_pos = (mov_b[2] - mov_a[2]) - (mov_b[2] - time_clock)
                    mov_p = point_at_time_agenda(mov_a, mov_b, time_pos)
                else:
                    mov_p = ta_cur_mov.end_point()

                lon, lat = get_lonlat(mov_p)
                agenda[time_clock] = [time_clock, lat, lon, 'move']
                # print lat, lon

        elif is_moving and time_clock < time_stop:

            if cur_mov_idx < len(ta_cur_mov):

                while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                    cur_mov_idx += 1

                # print 'cur_mov_idx', cur_mov_idx

                if cur_mov_idx < len(cur_mov):
                    mov_a = ta_cur_mov.point_n(cur_mov_idx - 1)
                    mov_b = ta_cur_mov.point_n(cur_mov_idx)
                    time_pos = (mov_b[2] - mov_a[2]) - (mov_b[2] - time_clock)
                    mov_p = point_at_time_agenda(mov_a, mov_b, time_pos)
                else:
                    mov_p = ta_cur_mov.end_point()

                lon, lat = get_lonlat(mov_p)
                agenda[time_clock] = [time_clock, lat, lon, 'move']
                # print lat, lon
                # print 'Moving to (%s, %s)' % (lon, lat), datetime.timedelta(seconds=time_clock), is_moving

            else:
                is_moving = False
                cur_mov = None
                ta_cur_mov = None
                cur_mov_idx = 1

                lon, lat = get_lonlat(loc_prototype[cur_lid])
                agenda[time_clock] = [time_clock, lat, lon, 'stop']
                # print lat, lon
                # print 'Arriving to (%s, %s)' % (lon, lat), datetime.timedelta(seconds=time_clock), is_moving
                # print ''

    ending_lid = cur_lid
    return agenda, ending_lid


def get_closest_lid_reinforcing(lon, lat, loc_prototype, loc_features):
    closest_lid = None
    min_dist = float('inf')

    for lid in loc_features:
        dist = spherical_distance(loc_prototype[lid], [lon, lat])
        if dist < min_dist:
            min_dist = dist
            closest_lid = lid

    return closest_lid


def get_closest_mid_reinforcing(lon, lat, mov_prototype, mid_lft, loc_features, cur_lid):
    closest_mid = None
    min_dist = float('inf')

    for mid in mov_prototype:

        for i in range(0, len(mov_prototype[mid]) - 2):
            dist1 = spherical_distance(mov_prototype[mid].point_n(i), [lon, lat])
            dist2 = spherical_distance(mov_prototype[mid].point_n(i+1), [lon, lat])
            dist3 = spherical_distance(mov_prototype[mid].point_n(i+2), [lon, lat])

            dist = (dist1 + dist2 + dist3) / 3.0
            if dist < min_dist and mid_lft[mid][0] in loc_features and mid_lft[mid][1] in loc_features and \
                            mid_lft[mid][1] != cur_lid:
                min_dist = dist
                closest_mid = mid

    return closest_mid


def generate_myagenda_reinforcing(myagenda_model, real_agenda, reinforcing_time=1800, random_choice=False,
                      time_start=0, time_offset=300, time_stop=86400, starting_lid=None):

    agenda = dict()
    time_clock = time_start

    loc_prototype = myagenda_model['loc_prototype']
    loc_features = myagenda_model['loc_features']
    lft_mid = myagenda_model['lft_mid']
    mid_lft = dict((v, k) for k, v in lft_mid.iteritems())

    # print lft_mid
    # print ''
    # print mid_lft

    mov_prototype = myagenda_model['mov_prototype']
    mov_features = myagenda_model['mov_features']

    is_moving = False
    cur_mov = None
    cur_mid = None
    ta_cur_mov = None
    cur_mov_idx = 1
    imposed_moving = False

    if starting_lid is None:
        starting_lid = sorted(loc_features.items(), key=lambda x: x[1]['loc_support'], reverse=True)[0][0]

    cur_lid = starting_lid
    lon, lat = get_lonlat(loc_prototype[cur_lid])
    agenda[time_clock] = [time_clock, lat, lon, 'stop']

    while time_clock < time_stop:

        time_clock += time_offset

        if not is_moving and time_clock < time_stop:

            next_lid = get_next_location(cur_lid, time_clock, time_offset, time_stop, loc_features[cur_lid],
                                         random_choice)
            is_moving = not next_lid == cur_lid

            if not is_moving:
                lon, lat = get_lonlat(loc_prototype[cur_lid])
                agenda[time_clock] = [time_clock, lat, lon, 'stop']

                if time_clock % reinforcing_time == 0:
                    agenda[time_clock] = real_agenda[time_clock]
                    lon, lat = real_agenda[time_clock][1], real_agenda[time_clock][2]
                    if real_agenda[time_clock][3] == 'stop':
                        cur_lid = get_closest_lid_reinforcing(lon, lat, loc_prototype, loc_features)
                    else:
                        is_moving = True
                        imposed_moving = True
                        cur_mid = get_closest_mid_reinforcing(lon, lat, mov_prototype, mid_lft, loc_features, cur_lid)
                        cur_mov = mov_prototype[cur_mid]
                        next_lid = mid_lft[cur_mid][1]
                        cur_lid = next_lid

                        mov_duration = get_movement_duration(cur_mid, mov_features)
                        real_duration = cur_mov.duration() / 1000
                        ratio_dur = 1.0 * mov_duration / real_duration
                        # print mov_duration, real_duration, ratio_dur

                        ta_time_start = time_clock - time_offset
                        ta_cur_mov = get_temporal_aligned_traj(cur_mov, ta_time_start, ratio_dur)

                        while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                            cur_mov_idx += 1

            if is_moving:

                lon, lat = get_lonlat(loc_prototype[cur_lid])
                # print 'Leaving location (%s, %s)' % (lon, lat), datetime.timedelta(seconds=time_clock), is_moving

                # print lat, lon
                if not imposed_moving:
                    cur_mid = lft_mid[(cur_lid, next_lid)]
                    cur_mov = mov_prototype[lft_mid[(cur_lid, next_lid)]]
                    cur_lid = next_lid

                    mov_duration = get_movement_duration(cur_mid, mov_features)
                    real_duration = cur_mov.duration() / 1000
                    ratio_dur = 1.0 * mov_duration / real_duration
                    # print mov_duration, real_duration, ratio_dur

                    ta_time_start = time_clock - time_offset
                    ta_cur_mov = get_temporal_aligned_traj(cur_mov, ta_time_start, ratio_dur)

                while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                    cur_mov_idx += 1

                # print 'cur_mov_idx', cur_mov_idx

                if cur_mov_idx < len(cur_mov):
                    mov_a = ta_cur_mov.point_n(cur_mov_idx - 1)
                    mov_b = ta_cur_mov.point_n(cur_mov_idx)
                    time_pos = (mov_b[2] - mov_a[2]) - (mov_b[2] - time_clock)
                    mov_p = point_at_time_agenda(mov_a, mov_b, time_pos)
                else:
                    mov_p = ta_cur_mov.end_point()

                lon, lat = get_lonlat(mov_p)
                agenda[time_clock] = [time_clock, lat, lon, 'move']
                # print lat, lon

                if time_clock % reinforcing_time == 0:
                    agenda[time_clock] = real_agenda[time_clock]
                    lon, lat = real_agenda[time_clock][1], real_agenda[time_clock][2]
                    if real_agenda[time_clock][3] == 'stop':
                        cur_lid = get_closest_lid_reinforcing(lon, lat, loc_prototype, loc_features)
                        is_moving = False
                        imposed_moving = False
                        cur_mov = None
                        ta_cur_mov = None
                        cur_mov_idx = 1
                    else:
                        reinforcing_cur_mid = get_closest_mid_reinforcing(lon, lat, mov_prototype, mid_lft,
                                                                          loc_features, cur_lid)
                        if reinforcing_cur_mid != cur_mid:
                            cur_mid = reinforcing_cur_mid
                            cur_mov = mov_prototype[cur_mid]
                            next_lid = mid_lft[cur_mid][1]
                            cur_lid = next_lid

                            mov_duration = get_movement_duration(cur_mid, mov_features)
                            real_duration = cur_mov.duration() / 1000
                            ratio_dur = 1.0 * mov_duration / real_duration
                            # print mov_duration, real_duration, ratio_dur

                            ta_time_start = time_clock - time_offset
                            ta_cur_mov = get_temporal_aligned_traj(cur_mov, ta_time_start, ratio_dur)

                            while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                                cur_mov_idx += 1

        elif is_moving and time_clock < time_stop:

            if cur_mov_idx < len(ta_cur_mov):

                while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                    cur_mov_idx += 1

                # print 'cur_mov_idx', cur_mov_idx

                if cur_mov_idx < len(cur_mov):
                    mov_a = ta_cur_mov.point_n(cur_mov_idx - 1)
                    mov_b = ta_cur_mov.point_n(cur_mov_idx)
                    time_pos = (mov_b[2] - mov_a[2]) - (mov_b[2] - time_clock)
                    mov_p = point_at_time_agenda(mov_a, mov_b, time_pos)
                else:
                    mov_p = ta_cur_mov.end_point()

                lon, lat = get_lonlat(mov_p)
                agenda[time_clock] = [time_clock, lat, lon, 'move']
                # print lat, lon
                # print 'Moving to (%s, %s)' % (lon, lat), datetime.timedelta(seconds=time_clock), is_moving

                if time_clock % reinforcing_time == 0:
                    agenda[time_clock] = real_agenda[time_clock]
                    lon, lat = real_agenda[time_clock][1], real_agenda[time_clock][2]
                    if real_agenda[time_clock][3] == 'stop':
                        cur_lid = get_closest_lid_reinforcing(lon, lat, loc_prototype, loc_features)
                        is_moving = False
                        imposed_moving = False
                        cur_mov = None
                        ta_cur_mov = None
                        cur_mov_idx = 1
                    else:
                        reinforcing_cur_mid = get_closest_mid_reinforcing(lon, lat, mov_prototype, mid_lft,
                                                                          loc_features, cur_lid)
                        if reinforcing_cur_mid != cur_mid:
                            cur_mid = reinforcing_cur_mid
                            cur_mov = mov_prototype[cur_mid]
                            next_lid = mid_lft[cur_mid][1]
                            cur_lid = next_lid

                            mov_duration = get_movement_duration(cur_mid, mov_features)
                            real_duration = cur_mov.duration() / 1000
                            ratio_dur = 1.0 * mov_duration / real_duration
                            # print mov_duration, real_duration, ratio_dur

                            ta_time_start = time_clock - time_offset
                            ta_cur_mov = get_temporal_aligned_traj(cur_mov, ta_time_start, ratio_dur)

                            while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                                cur_mov_idx += 1

            else:
                is_moving = False
                imposed_moving = False
                cur_mov = None
                ta_cur_mov = None
                cur_mov_idx = 1

                lon, lat = get_lonlat(loc_prototype[cur_lid])
                agenda[time_clock] = [time_clock, lat, lon, 'stop']

                if time_clock % reinforcing_time == 0:
                    agenda[time_clock] = real_agenda[time_clock]
                    lon, lat = real_agenda[time_clock][1], real_agenda[time_clock][2]
                    if real_agenda[time_clock][3] == 'stop':
                        cur_lid = get_closest_lid_reinforcing(lon, lat, loc_prototype, loc_features)
                    else:
                        is_moving = True
                        imposed_moving = True
                        cur_mid = get_closest_mid_reinforcing(lon, lat, mov_prototype, mid_lft, loc_features, cur_lid)
                        cur_mov = mov_prototype[cur_mid]
                        next_lid = mid_lft[cur_mid][1]
                        cur_lid = next_lid

                        mov_duration = get_movement_duration(cur_mid, mov_features)
                        real_duration = cur_mov.duration() / 1000
                        ratio_dur = 1.0 * mov_duration / real_duration
                        # print mov_duration, real_duration, ratio_dur

                        ta_time_start = time_clock - time_offset
                        ta_cur_mov = get_temporal_aligned_traj(cur_mov, ta_time_start, ratio_dur)

                        while cur_mov_idx < len(ta_cur_mov) and ta_cur_mov.point_n(cur_mov_idx)[2] < time_clock:
                            cur_mov_idx += 1

                # print lat, lon
                # print 'Arriving to (%s, %s)' % (lon, lat), datetime.timedelta(seconds=time_clock), is_moving
                # print ''

    ending_lid = cur_lid
    return agenda, ending_lid


def main():

    params = rome_params

    con = get_connection()
    cur = con.cursor()

    input_table = params['input_table']
    min_lat = params['min_lat']
    min_lon = params['min_lon']
    tzoffset = params['tzoffset']

    uid = 438563  # old_rome146099 # rome
    #uid = 659447  # london
    week_limit = 42
    traintest_date = datetime.datetime.strptime('2015-05-03', '%Y-%m-%d')

    imh = load_individual_mobility_history(cur, uid, input_table)

    history_order_dict = get_ordered_history(imh)

    # train, test = train_test_partition_weeklimit(imh, history_order_dict, week_limit)
    train, test, test_days = train_test_partition_date(imh, history_order_dict, date=traintest_date)

    print datetime.datetime.now()
    myagenda_model = build_myagenda(train, tzoffset)
    print datetime.datetime.now()

    myagenda, _ = generate_myagenda(myagenda_model, time_start=0, time_offset=300, time_stop=86400)

    print '------'

    for event_time in sorted(myagenda):
        event = myagenda[event_time]
        print datetime.timedelta(seconds=event_time), event[1], event[2], event[3]

if __name__ == "__main__":
    main()
