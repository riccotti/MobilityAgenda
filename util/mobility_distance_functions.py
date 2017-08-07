import math

from method.trajectory import *

__author__ = 'Riccardo Guidotti'


def spherical_distance(a, b):
    lat1 = a[1]
    lon1 = a[0]
    lat2 = b[1]
    lon2 = b[0]
    R = 6371000
    rlon1 = lon1 * math.pi / 180.0
    rlon2 = lon2 * math.pi / 180.0
    rlat1 = lat1 * math.pi / 180.0
    rlat2 = lat2 * math.pi / 180.0
    dlon = (rlon1 - rlon2) / 2.0
    dlat = (rlat1 - rlat2) / 2.0
    lat12 = (rlat1 + rlat2) / 2.0
    sindlat = math.sin(dlat)
    sindlon = math.sin(dlon)
    cosdlon = math.cos(dlon)
    coslat12 = math.cos(lat12)
    f = sindlat * sindlat * cosdlon * cosdlon + sindlon * sindlon * coslat12 * coslat12
    f = math.sqrt(f)
    f = math.asin(f) * 2.0 # the angle between the points
    f *= R
    return f


def start_end_distance(tr1, tr2):

    start1 = tr1.start_point()
    start2 = tr2.start_point()

    end1 = tr1.end_point()
    end2 = tr2.end_point()

    dist_start = spherical_distance(start1, start2)
    dist_end = spherical_distance(end1, end2)

    dist = dist_start + dist_end
    return dist


def calculate_traj_approximation(traj1, traj2, pred_thr, last_prop, time_mod=86400):
    """
    Calculate the approximation between the traejctory and the routine.
    """
    res = {
        'id_t1': traj1.id,
        'id_t2': traj2.id,
        'head': None,
        'tail': None,
        'dist': float('infinity')
    }

    # lool for the closest point on routine to the last point of traj
    t_last = traj1.end_point()
    min_dist = float('infinity')
    id_min = None
    for i in range(0, len(traj2) - 1):
        p = traj2.point_n(i)
        dist = spherical_distance(p, t_last)
        if dist < min_dist:
            min_dist = dist
            id_min = i

    cp = closest_point_on_segment(traj2.point_n(id_min), traj2.point_n(id_min + 1), t_last)

    # calcualte the distance between the two closest points
    dist = spherical_distance(cp, t_last)
    if last_prop == 0.0 and dist >= pred_thr:
        return res

    # cut the routine temporally from the beginning of traj to the time of the closest point
    t2 = cp[2] / 1000 % time_mod
    t1 = traj2.start_point()[2] / 1000 % time_mod
    traj2_cut = get_sub_trajectory(traj2, t1, t2)
    if traj2_cut is None or len(traj2_cut) < 3:
        return res

    # if the trajectory is shorter than the routine cut remove the initial part of the routine_cut
    if traj1.length() < traj2_cut.length():
        traj2_cut = get_sub_trajectory_keep_end(traj2_cut, traj1.length())

    # calculate the tail
    traj2_head = traj2_cut

    traj2_tail = get_sub_trajectory(traj2, t2, traj2.end_point()[2] / 1000 % time_mod)

    if traj2_tail is None:
        traj2_tail = Trajectory(id=traj2.id, object=[traj2.end_point()], vehicle=traj2.vehicle)

    traj2_tail.object.insert(0, traj2_cut.end_point())

    if last_prop > 0.0:
        last_traj = get_sub_trajectory_keep_end(traj1, traj1.length() * last_prop)
        last_routine_head = get_sub_trajectory_keep_end(traj2_head, traj2_head.length() * last_prop)
        if len(last_traj) >= 2 and len(last_routine_head) >= 2:
            dist = trajectory_distance(last_traj, last_routine_head)
            if dist >= pred_thr:
                return res

    res['head'] = traj2_head
    res['tail'] = traj2_tail
    res['dist'] = dist

    return res


def get_sub_trajectory(traj, from_ts, to_ts, time_mod=86400):
    """
    Cut traj according to the temporal thresholds from and to.
    """
    t_start = traj.start_point()[2] / 1000 % time_mod
    t_end = traj.end_point()[2] / 1000 % time_mod

    if to_ts < t_start:
        return None

    if from_ts > t_end:
        return None

    if from_ts < t_start:
        from_ts = t_start

    if to_ts > t_end:
        to_ts = t_end

    id_sub = traj.id
    object_sub = list()
    vehicle_sub = traj.vehicle

    for i in range(0, len(traj)):
        ts = traj.point_n(i)[2] / 1000 % time_mod
        if from_ts <= ts <= to_ts:
            object_sub.append(traj.point_n(i))
        if ts > to_ts:
            break

    sub_trajectory = Trajectory(id=id_sub, object=object_sub, vehicle=vehicle_sub)
    return sub_trajectory


def get_sub_trajectory_keep_end(traj, length):
    """
    Cut initial part of traj such that the length is respected.
    """
    if length >= traj.length():
        return traj

    id_sub = traj.id
    object_sub = [traj.end_point()]
    vehicle_sub = traj.vehicle

    tmp_length = 0
    for i in range(len(traj) - 1, 0, -1):
        p = traj.point_n(i)
        q = object_sub[len(object_sub) - 1]
        tmp_length += spherical_distance(q, p)
        if tmp_length >= length:
            break
        object_sub.insert(0, p)

    sub_trajectory = Trajectory(id=id_sub, object=object_sub, vehicle=vehicle_sub)
    return sub_trajectory


def point_at_time_agenda(a, b, ts):
    """
    Returns the points p at time ts between the points a and b
    """

    time_dist_a_b = b[2] - a[2]
    time_dist_a_p = ts

    # print time_dist_a_b, time_dist_a_p, '<<<<<'

    if time_dist_a_p >= time_dist_a_b:
        return b

    # find the distance from a to p
    # space_dist_a_b = spherical_distance(a, b)
    space_dist_a_b = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    space_dist_a_p = 1.0 * time_dist_a_p / time_dist_a_b * space_dist_a_b

    # find point p
    p = [0, 0, a[2] + ts]

    if b[0] - a[0] == 0:
        return b

    m = (b[1] - a[1]) / (b[0] - a[0])
    p_0_1 = a[0] + space_dist_a_p / math.sqrt(1 + m**2)
    p_0_2 = a[0] - space_dist_a_p / math.sqrt(1 + m**2)
    p[0] = p_0_1 if p_0_1 > a[0] else p_0_2
    p[1] = m * (p[0] - a[0]) + a[1]

    return p


def point_at_time(a, b, ts, time_mod=86400):
    """
    Returns the points p at time ts between the points a and b
    """

    time_dist_a_b = (b[2] / 1000) % time_mod - (a[2] / 1000) % time_mod
    time_dist_a_p = ts

    # print (b[2] / 1000) % time_mod, (a[2] / 1000) % time_mod, ts
    # print time_dist_a_b, time_dist_a_p, '<<<<<'

    if time_dist_a_p >= time_dist_a_b:
        # print 'QUI'
        return b

    # find the distance from a to p
    # space_dist_a_b = spherical_distance(a, b)
    space_dist_a_b = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    space_dist_a_p = 1.0 * time_dist_a_p / time_dist_a_b * space_dist_a_b

    # find point p
    p = [0, 0, a[2] + ts * 1000]

    if b[0] - a[0] == 0:
        # print 'QUO'
        return b

    m = (b[1] - a[1]) / (b[0] - a[0])
    p_0_1 = a[0] + space_dist_a_p / math.sqrt(1 + m**2)
    p_0_2 = a[0] - space_dist_a_p / math.sqrt(1 + m**2)
    p[0] = p_0_1 if p_0_1 > a[0] else p_0_2
    p[1] = m * (p[0] - a[0]) + a[1]

    return p


def __is_synch(p1, p2, time_th, time_mod=86400):
    ts1 = p1[2]/1000 % time_mod
    ts2 = p2[2]/1000 % time_mod
    return abs(ts1-ts2) >= time_th


def trajectory_distance_synch(tr1, tr2, time_th):
    if __is_synch(tr1.start_point(), tr2.start_point(), time_th) \
            and __is_synch(tr1.end_point(), tr2.end_point(), time_th):
        return float('infinity')
    else:
        return trajectory_distance(tr1, tr2)


def trajectory_distance_start_synch(tr1, tr2, time_th):
    if __is_synch(tr1.start_point(), tr2.start_point(), time_th):
        return float('infinity')
    else:
        return trajectory_distance(tr1, tr2)


def trajectory_distance_end_synch(tr1, tr2, time_th):
    if __is_synch(tr1.end_point(), tr2.end_point(), time_th):
        return float('infinity')
    else:
        return trajectory_distance(tr1, tr2)


def start_end_distance_synch(tr1, tr2, time_th):
    if __is_synch(tr1.start_point(), tr2.start_point(), time_th) \
            and __is_synch(tr1.end_point(), tr2.end_point(), time_th):
        return float('infinity')
    else:
        return start_end_distance(tr1, tr2)


def start_end_distance_start_synch(tr1, tr2, time_th):
    if __is_synch(tr1.start_point(), tr2.start_point(), time_th):
        return float('infinity')
    else:
        return start_end_distance(tr1, tr2)


def start_end_distance_end_synch(tr1, tr2, time_th):
    if __is_synch(tr1.end_point(), tr2.end_point(), time_th):
        return float('infinity')
    else:
        return start_end_distance(tr1, tr2)


def start_distance_synch(tr1, tr2, time_th):
    if __is_synch(tr1.start_point(), tr2.start_point(), time_th):
        return float('infinity')
    else:
        return start_distance(tr1, tr2)


def end_distance_synch(tr1, tr2, time_th):
    if __is_synch(tr1.end_point(), tr2.end_point(), time_th):
        return float('infinity')
    else:
        return end_distance(tr1, tr2)


def start_distance(tr1, tr2):

    start1 = tr1.start_point()
    start2 = tr2.start_point()

    dist_start = spherical_distance(start1, start2)

    dist = dist_start
    return dist


def end_distance(tr1, tr2):

    end1 = tr1.end_point()
    end2 = tr2.end_point()

    dist_end = spherical_distance(end1, end2)

    dist = dist_end
    return dist


def trajectory_distance(tr1, tr2):

    i1 = 0
    i2 = 0
    np = 0

    last_tr1 = tr1.point_n(i1)
    last_tr2 = tr2.point_n(i2)

    dist = spherical_distance(last_tr1, last_tr2)
    np += 1

    while True:

        step_tr1 = spherical_distance(last_tr1, tr1.point_n(i1+1))
        step_tr2 = spherical_distance(last_tr2, tr2.point_n(i2+1))

        if step_tr1 < step_tr2:
            i1 += 1
            last_tr1 = tr1.point_n(i1)
            last_tr2 = closest_point_on_segment(last_tr2, tr2.point_n(i2+1), last_tr1)
        elif step_tr1 > step_tr2:
            i2 += 1
            last_tr2 = tr2.point_n(i2)
            last_tr1 = closest_point_on_segment(last_tr1, tr1.point_n(i1+1), last_tr2)
        else:
            i1 += 1
            i2 += 1
            last_tr1 = tr1.point_n(i1)
            last_tr2 = tr2.point_n(i2)

        d = spherical_distance(last_tr1, last_tr2)

        dist += d
        np += 1

        if i1 >= (len(tr1)-1) or i2 >= (len(tr2)-1):
            break

    for i in range(i1, len(tr1)):
        d = spherical_distance(tr2.end_point(), tr1.point_n(i))
        dist += d
        np += 1

    for i in range(i2, len(tr2)):
        d = spherical_distance(tr1.end_point(), tr2.point_n(i))
        dist += d
        np += 1

    dist = 1.0 * dist / np

    return dist


def trajectory_distance2(tr1, tr2):

    i1 = 0
    i2 = 0
    np = 0

    last_tr1 = tr1.point_n(i1)
    last_tr2 = tr2.point_n(i2)

    tr1_length = tr1.length()
    tr2_length = tr2.length()

    tr_long = None
    tr_short = None
    if tr1_length <= tr2_length:
        """???"""

    dist = spherical_distance(last_tr1, last_tr2)
    np += 1

    while True:

        step_tr1 = spherical_distance(last_tr1, tr1.point_n(i1+1))
        step_tr2 = spherical_distance(last_tr2, tr2.point_n(i2+1))

        if step_tr1 < step_tr2:
            i1 += 1
            last_tr1 = tr1.point_n(i1)
            last_tr2 = closest_point_on_segment(last_tr2, tr2.point_n(i2+1), last_tr1)
        elif step_tr1 > step_tr2:
            i2 += 1
            last_tr2 = tr2.point_n(i2)
            last_tr1 = closest_point_on_segment(last_tr1, tr1.point_n(i1+1), last_tr2)
        else:
            i1 += 1
            i2 += 1
            last_tr1 = tr1.point_n(i1)
            last_tr2 = tr2.point_n(i2)

        d = spherical_distance(last_tr1, last_tr2)

        dist += d
        np += 1

        if i1 >= (len(tr1)-1) or i2 >= (len(tr2)-1):
            break

    for i in range(i1, len(tr1)):
        d = spherical_distance(tr2.end_point(), tr1.point_n(i))
        dist += d
        np += 1

    for i in range(i2, len(tr2)):
        d = spherical_distance(tr1.end_point(), tr2.point_n(i))
        dist += d
        np += 1

    dist = 1.0 * dist / np

    return dist


def closest_point_on_segment(a, b, p):
    sx1 = a[0]
    sx2 = b[0]
    sy1 = a[1]
    sy2 = b[1]
    sz1 = a[2]
    sz2 = b[2]
    px = p[0]
    py = p[1]

    x_delta = sx2 - sx1
    y_delta = sy2 - sy1
    z_delta = sz2 - sz1

    if x_delta == 0 and y_delta == 0:
        return p

    u = ((px - sx1) * x_delta + (py - sy1) * y_delta) / (x_delta * x_delta + y_delta * y_delta)
    if u < 0:
        closest_point = a
    elif u > 1:
        closest_point = b
    else:
        cp_x = sx1 + u * x_delta
        cp_y = sy1 + u * y_delta
        dist_a_cp = spherical_distance(a, [cp_x, cp_y, 0])
        if dist_a_cp != 0:
            cp_z = sz1 + long(z_delta / (spherical_distance(a, b) / spherical_distance(a, [cp_x, cp_y, 0])))
        else:
            cp_z = a[2]
        closest_point = [cp_x, cp_y, cp_z]

    return closest_point


def inclusion(tr1, tr2, space_th):
    """Return the sum of the distance between the two closest points of tr1 with the first and last points of tr2,
    check if tr2 is contained in tr1.
    """
    tr2_length = tr2.length()
    if tr2_length <= space_th:
        return float('infinity')

    start2 = tr2.start_point()
    end2 = tr2.end_point()

    i1_start2_point = None
    j1_end2_point = None
    i1_start2_dist = float('infinity')
    j1_end2_dist = float('infinity')

    i1 = 0
    j1 = 0

    for k in range(0, len(tr1)-1, 1):
        p1 = tr1.point_n(k)
        p2 = tr1.point_n(k+1)

        i1_start2_point_tmp = closest_point_on_segment(p1, p2, start2)
        i1_start2_dist_tmp = spherical_distance(start2, i1_start2_point_tmp)

        j1_end2_point_tmp = closest_point_on_segment(p1, p2, end2)
        j1_end2_dist_tmp = spherical_distance(end2, j1_end2_point_tmp)

        if i1_start2_dist_tmp < i1_start2_dist:
            i1_start2_dist = i1_start2_dist_tmp
            i1_start2_point = i1_start2_point_tmp
            i1 = k

        if j1_end2_dist_tmp < j1_end2_dist:
            j1_end2_dist = j1_end2_dist_tmp
            j1_end2_point = j1_end2_point_tmp
            j1 = k

    if None == i1_start2_point or None == j1_end2_point:
        return float('infinity')

    gap_i1_j1 = spherical_distance(i1_start2_point, j1_end2_point)

    if i1 >= j1 or gap_i1_j1 < space_th or (i1_start2_dist + j1_end2_dist) > tr2_length:
        return float('infinity')
    else:
        return i1_start2_dist + j1_end2_dist


def inclusion_synch(tr1, tr2, space_th, time_th, time_mod=86400):

    start1_time = tr1.start_point()[2]/1000 % time_mod
    end1_time = tr1.end_point()[2]/1000 % time_mod

    start2_time = tr2.start_point()[2]/1000 % time_mod
    end2_time = tr2.end_point()[2]/1000 % time_mod

    # tr2 last point (plus some wasting time) is before tr1 first point
    if end2_time + time_th < start1_time:
        return None

    # tr2 first point (minus some wasting time) is after tr1 last point
    if start2_time - time_th > end1_time:
        return None

    # tr2 first point (plus some wasting time) is before tr1 first point and consequently
    # also before any other points of tr1
    if start2_time + time_th < start1_time:
        return None

    # tr2 last point (minus some wasting time) is after tr1 last point and consequently
    # also after any other points of tr1
    if end2_time - time_th > end1_time:
        return None

    tr2_length = tr2.length()
    if tr2_length <= space_th:
        return None

    start2 = tr2.start_point()
    end2 = tr2.end_point()

    i1_start2_point = None
    j1_end2_point = None
    i1_start2_dist = float('infinity')
    j1_end2_dist = float('infinity')

    i1 = 0
    j1 = 0

    for k in range(0, len(tr1)-1, 1):
        p1 = tr1.point_n(k)
        p2 = tr1.point_n(k+1)

        i1_start2_point_tmp = closest_point_on_segment(p1, p2, start2)
        i1_start2_dist_tmp = spherical_distance(start2, i1_start2_point_tmp)
        i1_start2_time_diff = abs((start2[2]/1000 % time_mod) - (i1_start2_point_tmp[2]/1000 % time_mod))

        j1_end2_point_tmp = closest_point_on_segment(p1, p2, end2)
        j1_end2_dist_tmp = spherical_distance(end2, j1_end2_point_tmp)
        j1_end2_time_diff = abs((end2[2]/1000 % time_mod) - (j1_end2_point_tmp[2]/1000 % time_mod))

        if i1_start2_dist_tmp < i1_start2_dist and i1_start2_dist_tmp <= space_th/2.0 and i1_start2_time_diff <= time_th:
            i1_start2_dist = i1_start2_dist_tmp
            i1_start2_point = i1_start2_point_tmp
            i1 = k

        if j1_end2_dist_tmp < j1_end2_dist and j1_end2_dist_tmp <= space_th/2.0 and j1_end2_time_diff <= time_th:
            j1_end2_dist = j1_end2_dist_tmp
            j1_end2_point = j1_end2_point_tmp
            j1 = k

    if None == i1_start2_point or None == j1_end2_point:
        return None

    gap_i1_j1 = spherical_distance(i1_start2_point, j1_end2_point)

    if i1 >= j1 or gap_i1_j1 < space_th or (i1_start2_dist + j1_end2_dist) > tr2_length:
        return None
    else:
        i1_time = tr1.point_n(i1)[2]/1000 % time_mod
        j1_time = tr1.point_n(j1)[2]/1000 % time_mod

        match = {
            'space_dist_start_pickup': i1_start2_dist,
            'space_dist_end_drop_off': j1_end2_dist,
            'time_dist_start_pickup': abs(start2_time-i1_time),
            'time_dist_end_drop_off': abs(end2_time-j1_time),
            'time_pick_up_get_off': abs(i1_time-j1_time),
            'start_together': i1 == 0,
            'end_together': j1 == tr2.num_points()
        }

        return match

        # return i1_start2_dist + j1_end2_dist, abs(start2_time-i1_time) + abs(end2_time-j1_time), \
        #        i1 == 0, j1 == tr2.num_points(), \
        #        i1_start2_dist, j1_end2_dist, abs(start2_time-i1_time), abs(end2_time-j1_time)
