import psycopg2

from method.trajectory import *

__author__ = 'Riccardo Guidotti'


def get_connection():
    properties = {
        'dbname': 'dbname',
        'user': 'user',
        'host': 'host',
        'port': 'port',
        'password': 'password',
    }
    db_params = 'dbname=\'' + properties['dbname'] + '\' ' \
                'user=\'' + properties['user'] + '\' ' \
                'host=\'' + properties['host'] + '\' ' \
                'port=\'' + properties['port'] + '\' ' \
                'password=\'' + properties['password'] + '\''

    con = psycopg2.connect(db_params)

    return con


def extract_users_list(input_table, cur):
    query = """SELECT DISTINCT(uid) AS uid FROM %s""" % input_table
    cur.execute(query)
    rows = cur.fetchall()

    users = list()
    for r in rows:
        uid = str(r[0])
        users.append(uid)

    return sorted(users)


def load_individual_mobility_history(cur, uid, input_table):

    query = """SELECT id, ST_AsGeoJSON(object) AS object, uid
        FROM %s
        WHERE uid = '%s'""" % (input_table, uid)

    cur.execute(query)
    rows = cur.fetchall()
    trajectories = dict()

    for r in rows:
        trajectories[str(r[0])] = Trajectory(id=str(r[0]), object=json.loads(r[1])['coordinates'], vehicle=uid)

    imh = {'uid': uid, 'trajectories': trajectories}

    return imh


def load_mobility_histories(cur, users, input_table):

    users_str = '(\'%s\')' % ('\',\''.join(users))

    query = """SELECT id, ST_AsGeoJSON(object) AS object, uid
        FROM %s
        WHERE uid IN %s
        ORDER BY id""" % (input_table, users_str)

    cur.execute(query)
    rows = cur.fetchall()
    trajectories = dict()

    uid = None

    mobility_histories = dict()

    for r in rows:
        id = str(r[0])
        cur_uid = id.split('_')[0]

        if uid is None:
            uid = cur_uid

        if uid != cur_uid:
            trajectories[str(r[0])] = Trajectory(id=str(r[0]), object=json.loads(r[1])['coordinates'], vehicle=uid)
            mobility_histories[uid] = {'uid': uid, 'trajectories': trajectories}
            trajectories = dict()
            uid = cur_uid

        trajectories[str(r[0])] = Trajectory(id=str(r[0]), object=json.loads(r[1])['coordinates'], vehicle=uid)

    mobility_histories[uid] = {'uid': uid, 'trajectories': trajectories}

    return mobility_histories
