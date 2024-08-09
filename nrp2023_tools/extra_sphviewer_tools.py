import os
import numpy as np
import pandas
import h5py
from astropy import units as U
from simfiles import SimFiles
from simfiles.configs.APOSTLE import (
    __file__ as SFcfg,
    snapshots as snep_id_list,
    snap_id,
    snip_id,
)
from simobj import SimObj
from simobj.configs.APOSTLE import __file__ as SOcfg
import sphviewer
from simtrees import Tree, TreeTables
from simtrees.configs.APOSTLE import __file__ as STcfg
from collections import namedtuple
from scipy.interpolate import interp1d
from datetime import datetime
from .cosmic_time import t_to_a
from sphviewer.tools import camera_tools
import matplotlib.pyplot as plt
from matplotlib import cm

obj_id = namedtuple("obj_id", ["fof", "sub"])


def get_snepshots_and_times(t, sneplist, timelist, snipflags, snap_only=False):
    snap_mask = np.logical_not(snipflags) if snap_only else np.s_[...]
    index_after = np.searchsorted(timelist[snap_mask], t)
    index_before = index_after - 1
    if index_before < 0:
        index_before = 0
    if index_before >= len(sneplist[snap_mask]):
        index_before == len(sneplist[snap_mask]) - 1
    if index_after >= len(sneplist[snap_mask]):
        index_after == len(sneplist[snap_mask]) - 1

    return (
        sneplist[snap_mask][index_before],
        sneplist[snap_mask][index_after],
        snipflags[snap_mask][index_before],
        snipflags[snap_mask][index_after],
        timelist[snap_mask][index_before],
        timelist[snap_mask][index_after],
    )


def _db(
    snep,
    ptype="dm",
    extra_fields=tuple(),
    box_centre=None,
    box_halflength=None,
    fade_function=None,
):
    if (box_centre is not None) and (box_halflength is not None):
        boxmask = np.all(
            np.abs(snep[f"xyz_{ptype}"] - box_centre) < box_halflength, axis=1
        )
        dbsize = int(np.sum(boxmask))
    else:
        boxmask = np.s_[:]
        dbsize = snep[f"xyz_{ptype}"].shape[0]
    if len(snep["fields_from_snap"]) > 0:
        snap_db = np.recarray(
            (len(snep[f"snap_ids_{ptype}"]),),
            dtype=[
                ("id", np.int64),
            ]
            + [
                (extra_field, float)
                for extra_field in extra_fields
                if extra_field in snep["fields_from_snap"]
            ],
        )
        snap_db["id"] = snep[f"snap_ids_{ptype}"]
        for extra_field in extra_fields:
            if extra_field not in snep["fields_from_snap"]:
                continue
            snap_db[extra_field] = snep[f"{extra_field}_{ptype}"]
    db = np.recarray(
        (dbsize,),
        dtype=[
            ("id", np.int64),
            ("x", float),
            ("y", float),
            ("z", float),
            ("m", float),
        ]
        + [
            (extra_field, float)
            for extra_field in extra_fields
            if extra_field not in snep["fields_from_snap"]
        ],
    )
    db["id"] = snep[f"ids_{ptype}"][boxmask]
    db["x"] = snep[f"xyz_{ptype}"][:, 0][boxmask]
    db["y"] = snep[f"xyz_{ptype}"][:, 1][boxmask]
    db["z"] = snep[f"xyz_{ptype}"][:, 2][boxmask]
    db["m"] = snep[f"m_{ptype}"][boxmask]
    for extra_field in extra_fields:
        if extra_field in snep["fields_from_snap"]:
            continue
        db[extra_field] = snep[f"{extra_field}_{ptype}"][boxmask]
    db = fade_function(db)
    db = pandas.DataFrame(db)
    if len(snep["fields_from_snap"]) > 0:
        snap_db = pandas.DataFrame(snap_db)
        db = pandas.merge(db, snap_db, how="inner", on="id", suffixes=("", "_snap"))
    return db


class snep_interpolator(object):
    def __init__(self, res_phys_vol, fof_sub, use_snips=True, verbose=False, ncpu=1):
        # allow for 2 snepshots to be loaded at once
        self.res_phys_vol = res_phys_vol
        self.fof_sub = fof_sub
        self.earlier_snep = None
        self.later_snep = None
        self.db = None
        self.verbose = verbose
        self.ncpu = ncpu
        snaplist = [s[3] for s in snep_id_list if s[:3] == res_phys_vol and len(s) == 4]
        sniplist = [s[3] for s in snep_id_list if s[:3] == res_phys_vol and len(s) == 5]
        timelist = list()
        sneplist = list()
        snipflags = list()
        if self.verbose:
            print("checking available snapshot files:")
            print(snaplist)
        for count_s, s in enumerate(np.sort(snaplist), start=1):
            mysnap = snap_id(*res_phys_vol, snap=s)
            h5file = "{:s}/{:s}.0.hdf5".format(*snep_id_list[mysnap]["snapshot"])
            try:
                with h5py.File(h5file) as f:
                    a = f["Header"].attrs["Time"]
            except OSError:
                if self.verbose:
                    print(f"error, {h5file} not found")
                continue
            timelist.append(a)
            sneplist.append(s)
            snipflags.append(0)
        if use_snips:
            if self.verbose:
                print("checking available snipshot files:")
            for count_s, s in enumerate(np.sort(sniplist), start=1):
                mysnip = snip_id(*res_phys_vol, snip=s, is_snip=True)
                h5file = "{:s}/{:s}.0.hdf5".format(*snep_id_list[mysnip]["snapshot"])
                try:
                    with h5py.File(h5file) as f:
                        a = f["Header"].attrs["Time"]
                except OSError:
                    continue
                if a not in timelist:
                    # don't including snips at the same time as snaps (a=1, usually)
                    timelist.append(a)
                    sneplist.append(s)
                    snipflags.append(1)
        self.timelist = np.array(timelist)
        self.sneplist = np.array(sneplist)
        self.snipflags = np.array(snipflags)  # 0 if snap, 1 if snip
        isort = np.argsort(timelist)
        self.timelist = self.timelist[isort]
        self.sneplist = self.sneplist[isort]
        self.snipflags = self.snipflags[isort]

        return

    def _loadsnep(
        self,
        snep,
        snap,
        buffer="later",
        ptype="dm",
        is_snip=False,
        extra_fields=tuple(),
    ):
        if is_snip:
            mysnep = snip_id(*self.res_phys_vol, snip=snep, is_snip=True)
        else:
            mysnep = snap_id(*self.res_phys_vol, snap=snep)
        mysnap = snap_id(*self.res_phys_vol, snap=snap)
        SF = SimFiles(mysnep, configfile=SFcfg, ncpu=self.ncpu)
        SF_snap = SimFiles(mysnap, configfile=SFcfg, ncpu=self.ncpu)

        data = dict(snep=snep, ptype=ptype, is_snip=is_snip)
        SF.load(("a", "Lbox"), verbose=False)
        a = SF.a
        Lbox = SF.Lbox.to(U.kpc).value
        if ptype == "dm":
            t0 = datetime.now()
            SF.load(("xyz_dm", "ids_dm"), verbose=self.verbose)
            t1 = datetime.now()
            print(f"        reading xyz_dm, ids_dm for snep {snep} took:", t1 - t0)
            data["xyz_dm"] = SF.xyz_dm.to(U.kpc).value / a
            data["ids_dm"] = SF.ids_dm.value
            data["m_dm"] = np.ones(data["xyz_dm"].shape[0])
            data["xyz_dm"][data["xyz_dm"] < 0] += Lbox
        elif ptype == "g":
            t0 = datetime.now()
            SF.load(("xyz_g", "ids_g", "m_g"), verbose=self.verbose)
            t1 = datetime.now()
            print(f"        reading xyz_g, ids_g, m_g for snep {snep} took:", t1 - t0)
            data["xyz_g"] = SF.xyz_g.to(U.kpc).value / a
            data["ids_g"] = SF.ids_g.value
            data["m_g"] = SF.m_g.to(U.Msun).value
            data["xyz_g"][data["xyz_g"] < 0] += Lbox
        elif ptype == "s":
            t0 = datetime.now()
            SF.load(("xyz_s", "ids_s", "m_s"), verbose=self.verbose)
            t1 = datetime.now()
            print(f"        reading xyz_s, ids_s, m_s for snep {snep} took:", t1 - t0)
            data["xyz_s"] = SF.xyz_s.to(U.kpc).value / a
            data["ids_s"] = SF.ids_s.value
            data["m_s"] = SF.m_s.to(U.Msun).value
            data["xyz_s"][data["xyz_s"] < 0] += Lbox
        else:
            raise ValueError
        data["fields_from_snap"] = list()
        t0 = datetime.now()
        for extra_field in extra_fields:
            if hasattr(SF, f"{extra_field}_{ptype}"):
                data[f"{extra_field}_{ptype}"] = SF[f"{extra_field}_{ptype}"]
            else:
                try:
                    SF.load((f"{extra_field}_{ptype}",), verbose=self.verbose)
                except KeyError:
                    data["fields_from_snap"].append(extra_field)
                else:
                    data[f"{extra_field}_{ptype}"] = SF[f"{extra_field}_{ptype}"]
        if len(data["fields_from_snap"]) > 0:
            if not hasattr(SF_snap, f"ids_{ptype}"):
                SF_snap.load((f"ids_{ptype}",), verbose=self.verbose)
            data[f"snap_ids_{ptype}"] = SF_snap[f"ids_{ptype}"]
        for extra_field in data["fields_from_snap"]:
            if not hasattr(SF_snap, f"{extra_field}_{ptype}"):
                SF_snap.load((f"{extra_field}_{ptype}",), verbose=self.verbose)
            data[f"{extra_field}_{ptype}"] = SF_snap[f"{extra_field}_{ptype}"]
        if buffer == "earlier":
            self.earlier_snep = data
        elif buffer == "later":
            self.later_snep = data
        t1 = datetime.now()
        print(f"        reading extra_fields for snep {snep} took:", t1 - t0)
        return

    def _unload(self, buf):
        if buf == 0:
            del self.earlier_snep
            self.earlier_snep = None
        elif buf == 1:
            del self.later_snep
            self.later_snep = None
        else:
            raise ValueError
        return

    def __call__(
        self,
        t,
        ptype="dm",
        extra_fields=tuple(),
        box_centre=None,
        box_halflength=None,
        fade_function=None,
        unload=False,
    ):
        s0, s1, is0, is1, t0, t1 = get_snepshots_and_times(
            t, self.sneplist, self.timelist, self.snipflags
        )
        s0_snap, s1_snap, _, _, t0_snap, t1_snap = get_snepshots_and_times(
            t, self.sneplist, self.timelist, self.snipflags, snap_only=True
        )
        # check if already has in memory
        if (
            (self.earlier_snep is not None)
            and (self.later_snep is not None)
            and (self.earlier_snep["snep"] == s0)
            and (self.later_snep["snep"] == s1)
            and (self.earlier_snep["ptype"] == ptype)
            and (self.later_snep["ptype"] == ptype)
            and (self.earlier_snep["is_snip"] == is0)
            and (self.later_snep["is_snip"] == is1)
        ):
            db0 = _db(
                self.earlier_snep,
                ptype=ptype,
                extra_fields=extra_fields,
                box_centre=box_centre,
                box_halflength=box_halflength,
                fade_function=lambda pdata: fade_function(pdata, box_centre),
            )
            db1 = _db(
                self.later_snep,
                ptype=ptype,
                extra_fields=extra_fields,
                box_centre=box_centre,
                box_halflength=box_halflength,
                fade_function=lambda pdata: fade_function(pdata, box_centre),
            )
            if unload:
                self._unload(0)
                self._unload(1)
        else:
            if (
                (self.later_snep is not None)
                and (self.later_snep["snep"] == s0)
                and (self.later_snep["ptype"] == ptype)
                and (self.later_snep["is_snip"] == is0)
            ):
                self.earlier_snep = self.later_snep
                self._loadsnep(
                    s1,
                    s1_snap,
                    buffer="later",
                    ptype=ptype,
                    is_snip=is1,
                    extra_fields=extra_fields,
                )
                db0 = _db(
                    self.earlier_snep,
                    ptype=ptype,
                    extra_fields=extra_fields,
                    box_centre=box_centre,
                    box_halflength=box_halflength,
                    fade_function=lambda pdata: fade_function(pdata, box_centre),
                )
                db1 = _db(
                    self.later_snep,
                    ptype=ptype,
                    extra_fields=extra_fields,
                    box_centre=box_centre,
                    box_halflength=box_halflength,
                    fade_function=lambda pdata: fade_function(pdata, box_centre),
                )
                if unload:
                    self._unload(0)
                    self._unload(1)
            else:
                self._loadsnep(
                    s0,
                    s0_snap,
                    buffer="earlier",
                    ptype=ptype,
                    is_snip=is0,
                    extra_fields=extra_fields,
                )
                db0 = _db(
                    self.earlier_snep,
                    ptype=ptype,
                    extra_fields=extra_fields,
                    box_centre=box_centre,
                    box_halflength=box_halflength,
                    fade_function=lambda pdata: fade_function(pdata, box_centre),
                )
                if unload:
                    self._unload(0)
                self._loadsnep(
                    s1,
                    s1_snap,
                    buffer="later",
                    ptype=ptype,
                    is_snip=is1,
                    extra_fields=extra_fields,
                )
                db1 = _db(
                    self.later_snep,
                    ptype=ptype,
                    extra_fields=extra_fields,
                    box_centre=box_centre,
                    box_halflength=box_halflength,
                    fade_function=lambda pdata: fade_function(pdata, box_centre),
                )
                if unload:
                    self._unload(1)
            if self.verbose:
                print("Merging snapshots...", datetime.now())
            self.db = pandas.merge(
                db0, db1, how="inner", on="id", suffixes=("_earlier", "_later")
            )
            if self.verbose:
                print("Merge complete.", datetime.now())
        xyz = np.vstack(
            (self.db["x_earlier"], self.db["y_earlier"], self.db["z_earlier"])
        ).T + np.vstack(
            (
                self.db["x_later"] - self.db["x_earlier"],
                self.db["y_later"] - self.db["y_earlier"],
                self.db["z_later"] - self.db["z_earlier"],
            )
        ).T * (
            t - t0
        ) / (
            t1 - t0
        )

        interpolated_extra_fields = dict()
        for q in extra_fields:
            t0_xf = t0_snap if q in self.earlier_snep["fields_from_snap"] else t0
            t1_xf = t1_snap if q in self.later_snep["fields_from_snap"] else t1
            interpolated_extra_fields[q] = self.interpolate_scalar(q, t, t0_xf, t1_xf)
        return dict(
            xyz=xyz,
            m=self.interpolate_scalar("m", t, t0, t1),
            **interpolated_extra_fields,
        )

    def interpolate_scalar(self, q, t, t0, t1):
        return np.array(
            self.db[f"{q}_earlier"]
            + (self.db[f"{q}_later"] - self.db[f"{q}_earlier"]) * (t - t0) / (t1 - t0)
        )


class cam_interpolator(object):
    def __init__(self, res_phys_vol, fof_sub, cache_dir="cache"):
        try:
            os.mkdir(cache_dir)
        except FileExistsError:
            pass
        self.res_phys_vol = res_phys_vol
        self.t_cachename = "{:s}/tree_timelist_{:d}_{:s}_{:d}_{:d}_{:d}.npy".format(
            cache_dir, *res_phys_vol, *fof_sub
        )
        self.xyz_cachename = "{:s}/tree_xyzlist_{:d}_{:s}_{:d}_{:d}_{:d}.npy".format(
            cache_dir, *res_phys_vol, *fof_sub
        )
        self.id_cachename = "{:s}/tree_idlist_{:d}_{:s}_{:d}_{:d}_{:d}.npy".format(
            cache_dir, *res_phys_vol, *fof_sub
        )
        cache_loaded = None
        try:
            self.tree_timelist = np.load(self.t_cachename)
            self.tree_xyzlist = np.load(self.xyz_cachename)
            cache_loaded = True
        except IOError:
            cache_loaded = False
        if not cache_loaded:
            tt = TreeTables(snap_id(*res_phys_vol, snap=127), STcfg)
            tt.mass_filter(1e4 * U.Msun, particle_type=4)
            tree = Tree((127,) + fof_sub, treetables=tt)

            tree_timelist = list()
            tree_xyzlist = list()
            tree_idlist = list()

            print("Fetching coordinates")
            for node_count, node in enumerate(tree.trunk, start=1):
                snap, fof, sub = tt.sub_groups[node.key]
                print("{:d}/{:d}".format(node_count, len(tree.trunk)), end=", ")
                node_objid = obj_id(fof=fof, sub=sub)
                with SimObj(
                    obj_id=node_objid,
                    snap_id=snap_id(*res_phys_vol, snap=snap),
                    mask_type="fof",  # unused
                    mask_args=(node_objid,),
                    configfile=SOcfg,
                    simfiles_configfile=SFcfg,
                ) as SO:
                    tree_timelist.append(SO.a)
                    tree_xyzlist.append(SO.cops[0].to(U.kpc).value)
                tree_idlist.append([snap, fof, sub])
            tree_timelist = np.array(tree_timelist)[::-1]
            tree_xyzlist = np.array(tree_xyzlist)[::-1]
            tree_idlist = np.array(tree_idlist)[::-1]
            np.save(self.t_cachename, tree_timelist)
            np.save(self.xyz_cachename, tree_xyzlist)
            np.save(self.id_cachename, tree_idlist)
            self.tree_timelist = tree_timelist
            self.tree_xyzlist = tree_xyzlist
        self._interp = interp1d(
            self.tree_timelist,
            self.tree_xyzlist / self.tree_timelist[:, np.newaxis],
            axis=0,
            kind="cubic",
        )
        return

    def __call__(self, t):
        return self._interp(t)


def get_normalized_image(image, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(image)
    if vmax is None:
        vmax = np.max(image)

    image = np.clip(image, vmin, vmax)
    image = (image - vmin) / (vmax - vmin)

    return image


def boundary_v(frame, Nframes, boundary_v_early, boundary_v_late):
    return boundary_v_early + frame / (Nframes - 1) * (
        boundary_v_late - boundary_v_early
    )


def clean_angle_list(angle_list):
    angle_list = angle_list
    c = 0
    idx = np.where(angle_list != "pass")
    red_list = angle_list[idx]
    for i, angle in enumerate(red_list):
        if i == len(red_list) - 1:
            continue
        diff = float(red_list[i]) - float(red_list[i + 1])
        if abs(diff) > 180:
            if diff > 0:
                red_list[i + 1] = float(red_list[i + 1]) + 360
            elif diff < 0:
                red_list[i + 1] = float(red_list[i + 1]) - 360
    angle_list[idx] = red_list

    c = 0
    i = 1
    while angle_list[c] == "pass":
        if angle_list[c + i] != "pass":
            angle_list[c] = angle_list[c + i]
        i += 1

    c = -1
    while angle_list[c] == "pass":
        angle_list[c] = "same"
        c -= 1

    angle_list = angle_list.tolist()

    return angle_list


def init_tree_cache(res_phys_vol=None, fof_sub=None, save_dir="out", **kwargs):
    try:
        os.mkdir(f"{save_dir}")
    except FileExistsError:
        pass
    try:
        os.mkdir(f"{save_dir}/tree")
    except FileExistsError:
        pass
    cam_interpolator(res_phys_vol, fof_sub, cache_dir=f"{save_dir}/tree")
    return


def default_render(pdata, camera_location):
    if "hsm" in pdata.keys():
        P = sphviewer.Particles(pdata["xyz"], mass=pdata["m"], hsm=pdata["hsm"])
    else:
        P = sphviewer.Particles(pdata["xyz"], mass=pdata["m"])
    C = sphviewer.Camera()
    C.set_params(**camera_location)
    S = sphviewer.Scene(P, Camera=C)
    R = sphviewer.Render(S)
    R.set_logscale()
    img_arr = R.get_image()
    return (img_arr,)


def default_fade(pdata, centre, camera_location, fade_beyond=50):
    if (fade_beyond is not None) and (centre is not None):
        xyz = np.vstack((pdata["x"], pdata["y"], pdata["z"])).T
        r = np.sqrt(
            np.sum(
                np.power(xyz - centre, 2),
                axis=1,
            )
        )
        fade_mask = r > fade_beyond
        pdata["m"][fade_mask] = pdata["m"][fade_mask] * np.exp(
            -(r[fade_mask] / fade_beyond - 1)
        )
    return pdata


def make_frames_face_and_edge(
    res_phys_vol=None,
    fof_sub=None,
    ptype="g",
    extra_fields=tuple(),
    save_dir="out",
    render_function=default_render,
    fade_function=default_fade,
    r=100,
    npixels_x=1000,
    npixels_y=1000,
    start_time=1.5 * U.Gyr,
    end_time=13.759 * U.Gyr,
    Nframes=400,
    zoom=1,
    extent=None,
    segment=0,
    Nsegments=1,
    ncpu=1,
    overwrite=False,
    verbose=False,
    file_prefix=None,
):
    if file_prefix is None:
        file_prefix = ptype

    SI = snep_interpolator(res_phys_vol, fof_sub, ncpu=ncpu, verbose=verbose)
    CI = cam_interpolator(res_phys_vol, fof_sub, cache_dir=f"{save_dir}/tree/")
    anchors_face = {}
    anchors_face["sim_times"] = t_to_a(np.linspace(start_time, end_time, Nframes))
    anchors_face["id_frames"] = np.arange(Nframes)
    targets_face = CI(anchors_face["sim_times"])
    anchors_face["id_targets"] = np.arange(Nframes)
    anchors_face["r"] = [r] + ["same"] * (Nframes - 1)

    anchors_edge = {}
    anchors_edge["sim_times"] = t_to_a(np.linspace(start_time, end_time, Nframes))
    anchors_edge["id_frames"] = np.arange(Nframes)
    targets_edge = CI(anchors_edge["sim_times"])
    anchors_edge["id_targets"] = np.arange(Nframes)
    anchors_edge["r"] = [r] + ["same"] * (Nframes - 1)

    # creates empty lists for angles
    theta_list_face = np.empty(Nframes, dtype="U4")
    phi_list_face = np.empty(Nframes, dtype="U4")
    roll_list_face = np.empty(Nframes, dtype="U4")
    theta_list_edge = np.empty(Nframes, dtype="U4")
    phi_list_edge = np.empty(Nframes, dtype="U4")
    roll_list_edge = np.empty(Nframes, dtype="U4")

    for f in range(Nframes):
        theta_list_face[f], phi_list_face[f], roll_list_face[f] = np.load(
            f"{save_dir}/angles/tproll_face_{f:04d}.npy"
        )
        theta_list_edge[f], phi_list_edge[f], roll_list_edge[f] = np.load(
            f"{save_dir}/angles/tproll_edge_{f:04d}.npy"
        )

    # stops bound errors
    theta_list_face = clean_angle_list(theta_list_face)
    phi_list_face = clean_angle_list(phi_list_face)
    roll_list_face = clean_angle_list(roll_list_face)
    theta_list_edge = clean_angle_list(theta_list_edge)
    phi_list_edge = clean_angle_list(phi_list_edge)
    roll_list_edge = clean_angle_list(roll_list_edge)

    anchors_face["t"] = theta_list_face
    anchors_face["p"] = phi_list_face
    anchors_face["roll"] = roll_list_face
    anchors_face["zoom"] = [zoom] + ["same"] * (Nframes - 1)
    anchors_face["extent"] = [extent] + ["same"] * (Nframes - 1)

    anchors_edge["t"] = theta_list_edge
    anchors_edge["p"] = phi_list_edge
    anchors_edge["roll"] = roll_list_edge
    anchors_edge["zoom"] = [zoom] + ["same"] * (Nframes - 1)
    anchors_edge["extent"] = [extent] + ["same"] * (Nframes - 1)

    camera_trajectory = dict(
        face=camera_tools.get_camera_trajectory(targets_face, anchors_face),
        edge=camera_tools.get_camera_trajectory(targets_edge, anchors_edge),
    )

    start_seg_idx = segment * (
        Nframes // Nsegments
    )  # split into segments based on ncpus available
    end_seg_idx = (segment + 1) * (Nframes // Nsegments)

    if segment == Nsegments - 1:  # addition to ensure correct no. frames
        extra = (Nframes / Nsegments - Nframes // Nsegments) * Nsegments
        end_seg_idx += round(extra)

    for h in range(start_seg_idx, end_seg_idx):  # makes and saves the frame
        for alignment in ("face", "edge"):
            camera_location = camera_trajectory[alignment][h]
            camera_location["xsize"] = npixels_x
            camera_location["ysize"] = npixels_y
            img_arr_file = f"{save_dir}/{file_prefix}_{h:04d}_{alignment}"
            if not os.path.isfile(f"{img_arr_file}_0.npy") or overwrite:
                print(f"frame {h}, {alignment}-on ({ptype})", datetime.now())
                pdata = SI(  # calls __call__
                    camera_location["sim_times"],
                    ptype=ptype,
                    extra_fields=extra_fields,
                    box_centre=CI(camera_location["sim_times"]),
                    box_halflength=50 * 7,  # exp(-7)~1E-3
                    fade_function=lambda pdata, centre: fade_function(
                        pdata, centre, camera_location
                    ),
                )
                if verbose:
                    print("Starting rendering.", datetime.now())
                # SHOULD THIS BE `CI(camera_location)`??:
                img_arrs = render_function(pdata, camera_location)
                if verbose:
                    print("Rendering finished.", datetime.now())

                print(f"saving frame {h} {alignment}-on ({ptype})", datetime.now())
                for fnum, img_arr in enumerate(img_arrs):
                    np.save(f"{img_arr_file}_{fnum}.npy", img_arr)
            else:
                print(f"skipping {img_arr_file}, already exists")

    print("finished:", datetime.now())
    return camera_trajectory


def find_close_particles(N, coords):
    particle_radius = np.linalg.norm(
        coords, axis=1
    )  # find particle radii (cold gas only)
    sorted_radii = np.sort(particle_radius)
    close_radii = sorted_radii[0:N]  # closest N radii
    idx = np.empty(N).astype(int)  # empty list for indices
    for i, r in enumerate(close_radii):
        if len(np.where(particle_radius == r)[0]) == 1:
            idx[i] = int(
                np.where(particle_radius == r)[0]
            )  # find the index where the original list has radius r
    idx = np.sort(idx)  # closest particle indices
    return idx


def align_to_z(v):
    ident = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # identity matrix
    # rotation about x axis - theta
    v_yz = np.array([v[1], v[2]])
    Z_yz = np.array([0, 1])
    if np.linalg.norm(v_yz) != 0:  # if vector aligned with x axis, no rotation
        t = np.arccos(
            np.dot(v_yz, Z_yz) / (np.linalg.norm(v_yz) * np.linalg.norm(Z_yz))
        )  # find angle between in y-z plane (theta)
        if v[1] > 0:  # ensure correct rotation direction
            t = -t
        R_x = np.array(
            [
                [1, 0, 0],  # find rotation matrix
                [0, np.cos(t), np.sin(t)],
                [0, -np.sin(t), np.cos(t)],
            ]
        )
    else:
        R_x = ident

    v1 = np.dot(R_x, v)
    # rotation about y axis - phi
    v_xz = np.array([v1[0], v1[2]])
    Z_xz = Z_yz
    if np.linalg.norm(v_xz) != 0:  # if vector aligned with y axis, no rotation
        # find angle between in x-z plane (phi):
        p = np.arccos(
            np.dot(v_xz, Z_xz) / (np.linalg.norm(v_xz) * np.linalg.norm(Z_xz))
        )
        if v[0] < 0:  # ensure correct rotation direction
            p = -p
        R_y = np.array(
            [
                [np.cos(p), 0, -np.sin(p)],  # find rotation matric
                [0, 1, 0],
                [np.sin(p), 0, np.cos(p)],
            ]
        )
    else:
        R_y = ident
    roll = 0

    R = np.dot(R_y, R_x)
    t_p_roll = np.array([t, p, roll])

    return R, t_p_roll


def align_to_y(v):
    ident = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # identity matrix
    # rotation about X axis - theta
    v_yz = np.array([v[1], v[2]])
    Y_yz = np.array([1, 0])
    if np.linalg.norm(v_yz) != 0:  # if vector aligned with x axis, no rotation
        # find angle between in x-z plane (phi):
        t = np.arccos(
            np.dot(v_yz, Y_yz) / (np.linalg.norm(v_yz) * np.linalg.norm(Y_yz))
        )
        if v[2] < 0:  # ensure correct rotation direction
            t = -t
        R_x = np.array(
            [
                [1, 0, 0],  # find rotation matrix
                [0, np.cos(t), np.sin(t)],
                [0, -np.sin(t), np.cos(t)],
            ]
        )
    else:
        R_x = ident
        t = 0

    v1 = np.dot(R_x, v)
    # rotation about Z axis - roll
    v_yx = np.array([v1[0], v1[1]])
    Y_yx = np.array([0, 1])
    if np.linalg.norm(v_yx) != 0:  # if vector aligned with z axis, no rotation
        roll = np.arccos(
            np.dot(v_yx, Y_yx) / (np.linalg.norm(v_yx) * np.linalg.norm(Y_yx))
        )  # find angle between in y-x plane (roll)
        if v[0] > 0:  # ensure correct rotation direction
            roll = -roll
        R_z = np.array(
            [
                [np.cos(roll), np.sin(roll), 0],  # find rotation matrix
                [-np.sin(roll), np.cos(roll), 0],
                [0, 0, 1],
            ]
        )
    else:
        R_z = ident
        roll = 0

    R = np.dot(R_z, R_x)  # find total rotation, about x and then y
    p = 0
    if t < 0:
        t += 2 * np.pi
    if roll < 0:
        roll += 2 * np.pi
    t_p_roll = np.array([t, p, roll])
    return R, t_p_roll


def find_L_z_v2(SO):
    mask = SO.mHI_g / SO.m_g > 0.5  # selects only cold gas particles
    cold_gas_mass = SO.m_g[mask].to_value(
        U.solMass
    )  # making arrays dimensionless for further calcs
    cold_gas_coords = SO.xyz_g[mask].to_value(U.kpc)
    cold_gas_v = SO.vxyz_g[mask].to_value(U.km / U.s)

    if len(cold_gas_mass) > 250000:
        N_c = 250000 // 2
    else:
        N_c = len(cold_gas_mass) // 2
    closest = find_close_particles(N_c, cold_gas_coords)
    cold_gas_mass = cold_gas_mass[closest]
    cold_gas_coords = cold_gas_coords[closest]
    cold_gas_v = cold_gas_v[closest]

    closer = find_close_particles(100, cold_gas_coords)
    close_v = cold_gas_v[closer]
    zero_point_v = np.mean(close_v, axis=0)

    cold_gas_v = (
        cold_gas_v - zero_point_v
    )  # recentre velocity then continue with rotation

    cold_gas_momentum = cold_gas_mass[:, np.newaxis] * cold_gas_v

    gas_L_z = np.cross(cold_gas_coords, cold_gas_momentum)
    tot_L_z = np.sum(gas_L_z, axis=0)

    return tot_L_z, zero_point_v


def get_L_z_v0(obj, snap):
    SO = SimObj(
        obj_id=obj,
        snap_id=snap,
        mask_type="fofsub",
        mask_args=(obj,),
        configfile=SOcfg,
        simfiles_configfile=SFcfg,
    )
    # find total angular momentum and required rotation
    L_z, zero_point_v = find_L_z_v2(SO)
    return SO, L_z, zero_point_v


def find_snap_rotations(
    res_phys_vol=None,
    fof_sub=None,
    save_dir="out",
    start_time=1.5 * U.Gyr,
    end_time=13.759 * U.Gyr,
    Nframes=400,
    segment=0,
    Nsegments=1,
    ncpu=1,
    verbose=False,
    **kwargs,
):
    SI = snep_interpolator(res_phys_vol, fof_sub, ncpu=ncpu, verbose=verbose)
    obj_id = namedtuple("obj_id", ["fof", "sub"])
    tree_id = np.load(
        "{:s}/tree/tree_idlist_{:d}_{:s}_{:d}_{:d}_{:d}.npy".format(
            save_dir, *res_phys_vol, *fof_sub
        )
    )
    time_gyr = np.linspace(start_time, end_time, Nframes).to_value()
    times = t_to_a(np.linspace(start_time, end_time, Nframes))
    try:
        os.mkdir(f"{save_dir}/angles")
    except FileExistsError:
        pass
    np.save(f"{save_dir}/angles/frame_times.npy", time_gyr)
    start_seg_idx = segment * (Nframes // Nsegments)
    end_seg_idx = (segment + 1) * (Nframes // Nsegments)
    if segment == Nsegments - 1:
        extra = (Nframes / Nsegments - Nframes // Nsegments) * Nsegments
        end_seg_idx += round(extra)
    times_seg = times[start_seg_idx:end_seg_idx]
    sneps_before = []
    sneps_after = []
    for t in times_seg:
        s0, s1, is0, is1, t0, t1 = get_snepshots_and_times(
            t, SI.sneplist, SI.timelist, SI.snipflags
        )
        sneps_before.append(s0)  # finds required snep for times and frame (snep before)
        sneps_after.append(s1)
        # same for sneps - if s0 is snap and s1 snap, interpolate, if just 1 snap use it

    overwrite = False
    for f, snep in enumerate(sneps_after):  # for every required snep,
        tproll_face = np.empty(3, dtype="U4")
        tproll_edge = np.empty(3, dtype="U4")
        tproll_f_npy = "{:s}/angles/tproll_face_{:04d}.npy".format(
            save_dir, f + start_seg_idx
        )
        tproll_e_npy = "{:s}/angles/tproll_edge_{:04d}.npy".format(
            save_dir, f + start_seg_idx
        )
        i = np.where(SI.sneplist == snep)  # find index of the snep

        if (
            not os.path.isfile(tproll_f_npy) or overwrite
        ):  # if file doesn't exist, continue
            if SI.snipflags[i].all() == 1:  # if the snep is a snip
                tproll_face[0] = "pass"  # no cold gas data - smoothly interpolate
                tproll_face[1] = "pass"
                tproll_face[2] = "pass"
                tproll_edge[0] = "pass"
                tproll_edge[1] = "pass"
                tproll_edge[2] = "pass"
            else:  # if the snep is a snap, there is cold gas data
                # load the snap
                tree_idx = np.where(tree_id[:, 0] == snep)[0]
                tree_fof = tree_id[tree_idx, 1]
                tree_sub = tree_id[tree_idx, 2]
                mysnap = snap_id(*res_phys_vol, snap=snep)
                myobj = obj_id(fof=tree_fof, sub=tree_sub)
                SO, L_z, zero_point_v = get_L_z_v0(myobj, mysnap)
                rot_matrix_z, t_p_roll = align_to_z(L_z)  # add required angles to lists
                tproll_face[0] = t_p_roll[0] * 180 / np.pi
                tproll_face[1] = t_p_roll[1] * 180 / np.pi
                tproll_face[2] = t_p_roll[2] * 180 / np.pi

                rot_matrix_y, t_p_roll = align_to_y(L_z)
                tproll_edge[0] = t_p_roll[0] * 180 / np.pi
                tproll_edge[1] = t_p_roll[1] * 180 / np.pi
                tproll_edge[2] = t_p_roll[2] * 180 / np.pi

            np.save(tproll_f_npy, tproll_face)
            np.save(tproll_e_npy, tproll_edge)


def assemble_edge_and_face_video(Nframes=400, save_dir="out", ptype="g", **kwargs):
    # use face-on as reference for pixel values
    pixels_early = np.load(f"{save_dir}/{ptype}_{0:04d}_face_0.npy")
    pixels_late = np.load(f"{save_dir}/{ptype}_{Nframes - 2:04d}_face_0.npy")
    vmin_early = pixels_early.min()
    vmin_late = pixels_late.min()
    vmax_early = pixels_early.max()
    vmax_late = pixels_late.max()

    for n in range(Nframes):
        png_file = f"{save_dir}/{ptype}_{n:04d}.png"
        if os.path.exists(png_file):
            print(f"{png_file} already exists, skipping.")
            continue
        file_f = f"{save_dir}/{ptype}_{n:04d}_face_0.npy"
        file_e = f"{save_dir}/{ptype}_{n:04d}_edge_0.npy"
        pixels_f = np.load(file_f)
        pixels_e = np.load(file_e)
        rgb_f = cm.magma(
            get_normalized_image(
                pixels_f,
                vmin=boundary_v(n, Nframes - 1, vmin_early, vmin_late),
                vmax=boundary_v(n, Nframes - 1, vmax_early, vmax_late),
            )
        )
        rgb_e = cm.magma(
            get_normalized_image(
                pixels_e,
                vmin=boundary_v(n, Nframes - 1, vmin_early, vmin_late),
                vmax=boundary_v(n, Nframes - 1, vmax_early, vmax_late),
            )
        )
        rgb = np.hstack((rgb_f, rgb_e))
        plt.figure(figsize=(20, 10))
        plt.imshow(rgb)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.savefig(png_file, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
    os.system(
        "/cosma/local/ffmpeg/4.0.2/bin/ffmpeg"
        " -framerate 2"
        f" -i {save_dir}/{ptype}_%04d.png"
        " -y"
        " -c:v libx264"
        f" {save_dir}/{ptype}_edge_and_face.mp4"
    )
