"""
io.read.btsbioengineering

read specific BtsBioengineering file formats such as .tdf and .emt
extensions.

Functions
---------
read_tdf
    read the data contained in a tdf file.

read_emt
    read the data contained in a emt file.
"""

__all__ = ["read_tdf", "read_emt"]


#! IMPORTS


import struct
from io import BufferedReader
from typing import Any

import numpy as np
import pandas as pd

from ...utils import assert_file_extension

#! CONSTANTS


_BLOCK_KEYS = ["Type", "Format", "Offset", "Size"]


#! FUNCTIONS


def read_emt(
    path: str,
):
    """
    Return the readings from a .emt file as dicts of 3D objects.

    Parameters
    ----------
    path: str
        an existing tdf path.

    Returns
    -------
    emt: dict[str, pandas.DataFrame]
        a dict where each key is a specific 3D object contained in the emt file.

    object_type: str
        the type of data contained within the emt file.
    """

    # check the validity of the entered path
    assert_file_extension(path, "emt")

    # read the path
    with open(path, "r", encoding="utf-8") as buf:
        lines = [i.replace("\n", "").split("\t") for i in buf.readlines()]

    # get type and unit
    obj_unit = lines[3][1]

    # get headers, raw data and index
    headers = [tuple(i.rsplit(".", 1) + [obj_unit]) for i in lines[10][2:]]
    raw = np.array(lines[11:]).astype(float)
    index = pd.Index(raw[:, 1], name="Time [s]")
    names = ["LABEL", "DIMENSION", "UNIT"]
    cols = pd.MultiIndex.from_tuples(headers, names=names)
    data = pd.DataFrame(raw[:, 2:], index, cols)

    return data


def _get_label(
    obj: bytes,
):
    """
    _get_label convert a bytes string into a readable string

    Parameters
    ----------
    obj : bytes
        the bytes string to te read

    Returns
    -------
    label: str
        the decoded label
    """
    out = [chr(i) for i in struct.unpack("B" * len(obj), obj)]
    idx = np.where([i == chr(0) for i in out])[0]
    if len(idx) > 0:
        out = out[: idx[0]]
    return "".join(out).strip()


def _get_block(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
    block_id: int,
):
    """
    return the blocks according to the provided id

    Parameters
    ----------
    fid: BufferedReader
        the file stream object

    blocks : list[dict[str, int]]
        the blocks_info extracted by the _open_tdf function

    block_id : int
        the required block id

    Returns
    -------
    fid: BufferedReader
        the file stream object

    valid: dict[Literal["Type", "Format", "Offset", "Size"], int],
        the list of valid blocks
    """
    block = [i for i in blocks if block_id == i["Type"]]
    if len(block) > 0:
        block = block[0]
        fid.seek(block["Offset"], 0)
    else:
        block = {}
    return fid, block


def _read_frames_rts(
    fid: BufferedReader,
    nframes: int,
    cams: list[int] | np.ndarray[Any, np.dtype[np.int_]],
):
    """
    read frames from 2D data according to the RTS (real time stream) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    cams : list[int]
        the available cams

    Returns
    -------
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    frames = []
    ncams = len(cams)
    max_feats = 0

    # get the features
    for _ in np.arange(nframes):
        frame = []
        for _ in np.arange(ncams):
            nfeats = np.array(struct.unpack("i", fid.read(4)))[0]
            fid.seek(4, 1)
            vals = struct.unpack("f" * 2 * nfeats, fid.read(8 * nfeats))
            vals = np.reshape(vals, shape=(2, nfeats), order="F").T
            max_feats = max(max_feats, vals.shape[0])
            frame += [np.reshape(vals, shape=(2, nfeats), order="F").T]
        frames += [frame]

    # arrange as numpy array
    feats = np.ones((nframes, ncams, max_feats, 2)) * np.nan
    for frm, frame in enumerate(frames):
        for cam, feat in enumerate(frame):
            feats[frm, cam, np.arange(feat.shape[0]), :] = feat

    return feats.astype(np.float32)


def _read_frames_pck(
    fid: BufferedReader,
    nframes: int,
    cams: list[int] | np.ndarray[Any, np.dtype[np.int_]],
):
    """
    read frames from 2D data according to the PCK (packed data) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    cams : list[int]
        the available cams

    Returns
    -------
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    ncams = len(cams)
    nsamp = int(ncams * nframes)
    nfeats = struct.unpack(f"{nsamp}h", fid.read(2 * nsamp))
    nfeats = np.reshape(nfeats, shape=(ncams, nframes), order="F")
    max_feats = int(np.max(nfeats))
    feats = np.ones((nframes, ncams, max_feats, 2)) * np.nan
    for frm in np.arange(nframes):
        for cam in np.arange(ncams):
            num = int(2 * nfeats[cam, frm])
            vals = struct.unpack(f"{num}f", fid.read(4 * num))
            vals = np.reshape(vals, shape=(2, nfeats[cam, frm]), order="F").T
            feats[frm, cam, np.arange(vals.shape[0]), :] = vals
    return feats.astype(np.float32)


def _read_frames_syn(
    fid: BufferedReader,
    nframes: int,
    cams: list[int] | np.ndarray[Any, np.dtype[np.int_]],
):
    """
    read frames from 2D data according to the SYNC (synchronized data) format.

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        number of available frames

    cams : list[int]
        the available cams

    Returns
    -------
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    ncams = len(cams)
    max_feats = np.array(struct.unpack("1h", fid.read(2)))[0]
    shape = (nframes, ncams)
    nsamp = int(np.prod(shape))
    nfeats = struct.unpack(f"{nsamp}h", fid.read(2 * nsamp))
    nfeats = np.reshape(nfeats, shape=shape, order="F")
    feats = np.ones((nframes, ncams, max_feats, 2)) * np.nan
    for frm in np.arange(nframes):
        for cam in np.arange(ncams):
            nsamp = 2 * max_feats
            vals = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
            vals = np.reshape(vals, shape=(2, max_feats), order="F")
            vals = vals[:, : nfeats[cam, frm]].T
            feats[frm, cam, np.arange(vals.shape[0]), :] = vals
    return feats.astype(np.float32)


def _read_tracks(
    fid: BufferedReader,
    nframes: int,
    ntracks: int,
    nchannels: int,
    haslabels: bool,
):
    """
    read data by track

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        available frames

    ntracks : int
        number of tracks

    nchannels: int
        the number of channels to extract

    haslabels: bool
        if True the track labels are returned

    Returns
    -------
    frames: ndarray
        the features list with shape (nframes, ncams, nfeats, 2)
    """
    obj = np.ones((ntracks, nframes, nchannels), dtype=np.float32) * np.nan
    lbls = []
    for trk in np.arange(ntracks):
        # get the labels
        if haslabels:
            lbls += [_get_label(fid.read(256))]
        else:
            lbls += [f"track{trk + 1}"]

        # get the available segments
        nseg = np.array(struct.unpack("i", fid.read(4)))[0]
        fid.seek(4, 1)
        shape = (2, nseg)
        nsamp = int(np.prod(shape))
        segments = struct.unpack(f"{nsamp}i", fid.read(4 * nsamp))
        segments = np.reshape(segments, shape=shape, order="F").T

        # read the data for the actual track
        for start, stop in segments:
            for frm in np.arange(stop) + start:
                vals = fid.read(4 * nchannels)
                if frm < obj.shape[1]:
                    obj[trk, frm, :] = struct.unpack("f" * nchannels, vals)

    # split data by track
    return dict(zip(lbls, obj.astype(np.float32)))


def _read_frames(
    fid: BufferedReader,
    nframes: int,
    ntracks: int,
    nchannels: int,
    haslabels: bool = True,
):
    """
    read 3D data by frame

    Parameters
    ----------
    fid : BufferedReader
        file stream

    nframes : int
        available frames

    ntracks : int
        number of tracks

    nchannels: int
        the number of channels to extract

    haslabels: bool
        if True the track labels are returned

    Returns
    -------
    data3d: dict[str, pd.DataFrame]
        the parsed tracks.
    """

    # get the labels
    lbls = []
    for trk in np.arange(ntracks):
        if haslabels:
            label = _get_label(fid.read(256))
        else:
            label = f"track{trk + 1}"
        lbls += [label]

    # get the available data
    nsamp = nchannels * ntracks * nframes
    data = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
    data = np.reshape(data, shape=(ntracks, nframes, nchannels), order="F")
    data = data.astype(np.float32)

    # return
    return dict(zip(lbls, data))


def _read_camera_params(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read calibration data for general purpose

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    calibration_data: dict[str, Any]
        the available calibration data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 2)
    if len(block) == 0:
        return None
    cam_n = struct.unpack("i", fid.read(4))[0]  # number of cams
    cam_m = struct.unpack("I", fid.read(4))[0]  # model
    if cam_m < 4:
        cam_m = ["none", "kali", "amass", "thor"][cam_m]
    else:
        raise ValueError("cam_m value not recognized")
    cam_d = np.array(struct.unpack("3f", fid.read(12)), dtype=np.float32)
    cam_r = np.reshape(struct.unpack("9f", fid.read(36)), shape=(3, 3), order="F")
    cam_r = cam_r.astype(np.float32)
    cam_t = np.array(struct.unpack("3f", fid.read(12))).astype(np.float32)

    # channels map
    cam_map = struct.unpack(f"{cam_n}h", fid.read(2 * cam_n))

    # parameters
    cam_params = {}
    for i in cam_map:
        if 1 == block["Format"]:  # Seelab type 1 calibration
            params = {
                "R": np.reshape(
                    struct.unpack("9d", fid.read(72)), shape=(3, 3), order="F"
                ),
                "T": np.array(struct.unpack("3d", fid.read(24))),
                "F": np.array(struct.unpack("2d", fid.read(16))),
                "C": np.array(struct.unpack("2d", fid.read(16))),
                "K1": np.array(struct.unpack("2d", fid.read(16))),
                "K2": np.array(struct.unpack("2d", fid.read(16))),
                "K3": np.array(struct.unpack("2d", fid.read(16))),
                "VP": np.reshape(
                    struct.unpack("4i", fid.read(16)), shape=(2, 2), order="F"
                ),
                # origin = VP[:, 0] size = VP[:, 1]
            }
        elif 2 == block["Format"]:  # BTS
            params = {
                "R": np.reshape(
                    struct.unpack("9d", fid.read(72)), shape=(3, 3), order="F"
                ),
                "T": np.array(struct.unpack("3d", fid.read(24))),
                "F": np.array(struct.unpack("1d", fid.read(16))),
                "C": np.array(struct.unpack("2d", fid.read(16))),
                "KX": np.array(struct.unpack("70d", fid.read(560))),
                "KY": np.array(struct.unpack("70d", fid.read(560))),
                "VP": np.reshape(
                    struct.unpack("4i", fid.read(16)), shape=(2, 2), order="F"
                ),
                # origin = VP[:, 0] size = VP[:, 1]
            }
        else:
            msg = f"block['Format'] must be 1 or 2, but {block['Format']}"
            msg += " was found."
            raise ValueError(msg)
        params["VP"] = params["VP"].astype(np.int32)
        cam_params[i] = params

    return {
        "PARAMS": cam_params,
        "DIMENSION": cam_d,
        "ROTATION_MATRIX": cam_r,
        "TRANSLATION": cam_t,
        "MODEL": cam_m,
    }


def _read_camera_calibration(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read 2D data sequence from a tdf file stream to be used
    for camera calibration.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 3)
    if len(block) == 0:
        return None
    ncams, naxesframes, nwandframes, freq = struct.unpack("iiii", fid.read(16))
    fid.seek(4, 1)
    axespars = np.array(struct.unpack("9f", fid.read(36))).astype(np.float32)
    wandpars = np.array(struct.unpack("2f", fid.read(8))).astype(np.float32)

    # channels map
    cam_map = list(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))

    # features extraction function
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # get the features
    axesfeats = read_frames(fid, naxesframes, cam_map)
    wandfeats = read_frames(fid, nwandframes, cam_map)

    return {
        "AXES": {"FEATURES": axesfeats, "PARAMS": axespars},
        "WAND": {"FEATURES": wandfeats, "PARAMS": wandpars},
        "CHANNELS": cam_map,
        "FREQUENCY": freq,
    }


def _read_camera_raw(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read 2D data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 4)
    if len(block) == 0:
        return None
    ncams, nframes, freq, time0 = struct.unpack("iiif", fid.read(16))
    fid.seek(4, 1)

    # channels map
    cam_map = list(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))

    # features extraction
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # get the features
    return {
        "FEATURES": read_frames(fid, nframes, cam_map),
        "FREQUENCY": freq,
        "TIME0": time0,
        "CHANNELS": cam_map,
    }


def _read_camera_tracked(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read 2D data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 5)
    if len(block) == 0:
        return None
    nframes, freq, time0, ntracks = struct.unpack("iifi", fid.read(16))
    dims = np.array(struct.unpack("3f", fid.read(12)))
    rmat = np.reshape(struct.unpack("9f", fid.read(36)), shape=(3, 3), order="F")
    tras = np.array(struct.unpack("3f", fid.read(12)))
    fid.seek(4, 1)

    # get links
    if block["Format"] in [1, 3]:
        nlinks = struct.unpack("i", fid.read(4))[0]
        fid.seek(4, 1)
        nsamp = 2 * nlinks
        links = struct.unpack(f"{nsamp}i", fid.read(nsamp * 4))
        links = np.reshape(links, shape=(2, nlinks), order="F").T
    else:
        links = np.array([])

    # get the data
    if block["Format"] in [1, 2]:
        tracks = _read_tracks(fid, nframes, ntracks, 3, True)
    elif block["Format"] in [3, 4]:
        tracks = _read_frames(fid, nframes, ntracks, 3, True)
    else:
        msg = f"block['Format'] must be 1, 2, 3 or 4, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # convert the tracks in pandas dataframes
    idx = np.arange(list(tracks.values())[0].shape[0]) / freq + time0  # type: ignore
    idx = pd.Index(idx, name="TIME [s]")
    kys = list(tracks.keys())
    nms = ["LABEL", "DIMENSION", "UNIT"]
    col = pd.MultiIndex.from_product([kys, ["X", "Y", "Z"], ["m"]], names=nms)
    tracks = np.concatenate(list(tracks.values()), axis=1)
    tracks = pd.DataFrame(tracks, idx, col)

    # update the links with the names of the tracks
    links = np.array([[kys[i] for i in j] for j in links])

    return {
        "TRACKS": tracks,
        "LINKS": links,
        "DIMENSIONS": dims.astype(np.float32),
        "ROTATION_MATRIX": rmat.astype(np.float32),
        "TRASLATION": tras.astype(np.float32),
    }


def _read_camera_configuration(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read cameras physical configuration from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks : list[dict[Literal["Type", "Format", "Offset", "Size"], int]]
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 6)
    if len(block) == 0:
        return None
    nchns = struct.unpack("i", fid.read(4))[0]
    fid.seek(4, 1)

    # get the data
    cameras: dict[str, Any] = {}
    for _ in np.arange(nchns):
        logicalindex = struct.unpack("i", fid.read(4))[0]
        fid.seek(4, 1)
        lensname = _get_label(fid.read(32))
        camtype = _get_label(fid.read(32))
        camname = _get_label(fid.read(32))
        viewport = np.reshape(
            struct.unpack("i" * 4, fid.read(16)), shape=(2, 2), order="F"
        )
        cameras[camname] = {
            "INDEX": logicalindex,
            "TYPE": camtype,
            "LENS": lensname,
            "VIEWPORT": viewport.astype(np.int32),
        }

    return cameras


def _read_platforms_params(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read platforms calibration parameters from tdf file.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 7)
    if len(block) == 0:
        return None
    nplats = struct.unpack("i", fid.read(4))[0]
    fid.seek(4, 1)

    # channels map
    plat_map = np.array(struct.unpack(f"{nplats}h", fid.read(2 * nplats)))

    # read data for each platform
    platforms: list[dict[str, Any]] = []
    cols = pd.MultiIndex.from_product([["X", "Y", "Z"], ["m"]])
    for plat in plat_map:
        lbl = _get_label(fid.read(256))
        size = np.array(struct.unpack("ff", fid.read(8))).astype(np.float32)
        pos = np.reshape(struct.unpack("12f", fid.read(48)), shape=(3, 4), order="F").T
        platforms += [
            {
                "SIZE": size,
                "POSITION": pd.DataFrame(pos.astype(np.float32), columns=cols),
                "LABEL": lbl,
                "CHANNEL": plat,
            }
        ]
        fid.seek(256, 1)  # matlab code is missing this step

    return platforms


def _read_platforms_calibration(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read 2D data sequence from a tdf file stream to be used
    for force platforms calibration.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 8)
    if len(block) == 0:
        return None
    (
        nplats,
        ncams,
        freq,
    ) = struct.unpack("iii", fid.read(12))
    fid.seek(4, 1)

    # channels map
    cam_map = np.array(struct.unpack(f"{ncams}h", fid.read(2 * ncams)))
    plat_map = list(struct.unpack(f"{nplats}h", fid.read(2 * nplats)))

    # features extraction function
    if 1 == block["Format"]:  # RTS: Real Time Stream
        read_frames = _read_frames_rts
    elif 2 == block["Format"]:  # PCK: Packed Data format
        read_frames = _read_frames_pck
    elif 3 == block["Format"]:  # SYNC: Synchronized Data format
        read_frames = _read_frames_syn
    else:
        msg = f"block['Format'] must be 1, 2 or 3, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # read data for each platform
    platforms = {}
    for plt in plat_map:
        obj = {}
        label = _get_label(fid.read(256))
        frames = struct.unpack("i", fid.read(4))[0]
        obj["SIZE"] = np.array(struct.unpack("ff", fid.read(8)))
        obj["SIZE"] = obj["SIZE"].astype(np.float32)
        if frames != 0:
            obj["FEATURES"] = read_frames(fid, frames, cam_map)
            obj["LABEL"] = label
        else:
            obj["FEATURES"] = np.ones((frames, len(cam_map), 2, 0)) * np.nan
            obj["LABEL"] = ""

    return {
        "PLATFORMS": platforms,
        "FREQUENCY": freq,
        "CAMERA_CHANNELS": cam_map,
    }


def _read_platforms_raw(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read generic (untracked) platforms data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 9)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # channels data
    chn_map = np.array(struct.unpack("h" * ntracks, fid.read(2 * ntracks)))
    if block["Format"] in [1, 2, 3, 4]:
        nchns = 6
    elif block["Format"] in [5, 6, 7, 8]:
        nchns = 12
    else:
        msg = "block['Format'] must be a number in the 1-8 range, "
        msg += " was found."
        raise ValueError(msg)

    # has labels
    haslbls = block["Format"] in [3, 4, 7, 8]

    # get the data
    if block["Format"] in [1, 3, 5, 7]:
        tracks = _read_tracks(fid, nframes, ntracks, nchns, haslbls)
    else:  # i.e. block["Format"] in [2, 4, 6, 8]:
        # get the labels
        lbl = []
        for idx in np.arange(ntracks):
            if haslbls:
                lbl += [_get_label(fid.read(256))]
            else:
                lbl = [f"track{idx + 1}"]

        # get the available data
        nsamp = nchns * ntracks * nframes
        obj = struct.unpack(f"{nsamp}f", fid.read(4 * nsamp))
        obj = np.reshape(obj, shape=(nframes, ntracks, nchns), order="F")
        obj = np.transpose(obj, axes=[1, 0, 2]).astype(np.float32)
        tracks = dict(zip(lbl, obj))

    # get labels and units
    labels = ["ORIGIN.X", "ORIGIN.Y", "FORCE.X", "FORCE.Y", "FORCE.Z", "TORQUE"]
    units = ["m", "m", "N", "N", "N", "Nm"]
    if nchns == 12:
        labels = ["R." + i for i in labels] + ["L." + i for i in labels]
        units += units

    # convert the recovered data in 3D features having shape
    # (frames, track, nchns)
    feats = [np.expand_dims(i, 1) for i in tracks.values()]
    feats = np.concatenate(feats, axis=1)

    # return
    return {
        "FEATURES": feats,
        "CHANNELS": chn_map,
        "LABELS": labels,
        "UNITS": units,
        "TIME0": time0,
        "FREQUENCY": freq,
    }


def _get_muscle_name(raw_muscle_name: str):
    """
    private method used understand the muscle side according to its name

    Parameters
    ----------
    raw_muscle_name: str
        the raw muscle name

    Returns
    -------
    name: tuple[str, str | None]
        the muscle name divided as (<NAME>, <SIDE>). If proper side is not
        found (e.g. for Rectus Abdominis), the <SIDE> term is None.
    """
    # split the raw muscle name in words
    splits = raw_muscle_name.split(" ")

    # get the index of the word denoting the side
    side_idx = [i for i, v in enumerate(splits) if v in ["Left", "Right"]]
    side_idx = None if len(side_idx) == 0 else side_idx[0]

    # adjust the muscle name
    side = None if side_idx is None else splits.pop(side_idx)
    muscle = " ".join([i.capitalize() for i in splits[:2]])

    # return the tuple
    return (muscle, side)


def _read_emg(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read EMG data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 11)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # channels data
    chn_map = np.array(struct.unpack("h" * ntracks, fid.read(2 * ntracks)))

    # get the data
    if block["Format"] in [1]:
        tracks = _read_tracks(fid, nframes, ntracks, 1, True)
    elif block["Format"] in [2]:
        tracks = _read_frames(fid, nframes, ntracks, 1, True)
    else:
        msg = f"block['Format'] must be 1, 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # convert the tracks in a single pandas dataframe
    tracks = pd.DataFrame({i: v.flatten() for i, v in tracks.items()})
    cols = [_get_muscle_name(i) + ("V",) for i in list(tracks.keys())]
    tracks.columns = pd.MultiIndex.from_tuples(cols)
    idx = pd.Index(np.arange(tracks.shape[0]) / freq + time0, name="TIME [s]")
    tracks.index = idx
    return {
        "TRACKS": tracks.sort_index(axis=1),
        "EMG_CHANNELS": chn_map.astype(np.int16),
    }


def _read_platforms_tracked(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read force 3D data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 12)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # get the calibration data
    cald = np.array(struct.unpack("f" * 3, fid.read(12)))
    calr = np.reshape(struct.unpack("f" * 9, fid.read(36)), shape=(3, 3), order="F")
    calt = np.array(struct.unpack("f" * 3, fid.read(12)))
    fid.seek(4, 1)

    # get the data
    if block["Format"] in [1]:
        tracks = _read_tracks(fid, nframes, ntracks, 9, True)
    elif block["Format"] in [2]:
        tracks = _read_frames(fid, nframes, ntracks, 9, True)
    else:
        msg = f"block['Format'] must be 1, or 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # prepare the output data
    out = {
        "TRACKS": pd.DataFrame(),
        "DIMENSIONS": cald.astype(np.float32),
        "ROTATION_MATRIX": calr.astype(np.float32),
        "TRASLATION": calt.astype(np.float32),
    }

    # convert the tracks in pandas dataframes
    tarr = np.arange(list(tracks.values())[0].shape[0]) / freq + time0  # type: ignore
    tarr = pd.Index(tarr, name="TIME [s]")
    axs = ["X", "Y", "Z"]
    pairs = tuple(zip(["ORIGIN", "FORCE", "TORQUE"], ["m", "N", "Nm"]))
    nms = ["LABEL", "SOURCE", "DIMENSION", "UNIT"]
    objs = []
    for trk, arr in tracks.items():
        for idx, pair in enumerate(pairs):
            src, unt = pair
            dims = 3 * idx + np.arange(3)
            cols = [[trk], [src], axs, [unt]]
            cols = pd.MultiIndex.from_product(cols, names=nms)
            objs += [pd.DataFrame(arr[:, dims], index=tarr, columns=cols)]  # type: ignore
    out["TRACKS"] = pd.concat(objs, axis=1)

    return out


def _read_volume(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read volumetric data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 13)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # get the data
    if block["Format"] in [1]:  # by track
        tracks = {}
        for _ in np.arange(ntracks):
            label = _get_label(fid.read(256))

            # get the available segments
            nseg = np.array(struct.unpack("i", fid.read(4)))[0]
            fid.seek(4, 1)
            nsamp = 2 * nseg
            segments = struct.unpack(f"{nsamp}i", fid.read(4 * nsamp))
            segments = np.reshape(segments, shape=(2, nseg), order="F").astype(
                np.float32
            )

            # read the data for the actual track
            arr = np.ones((nframes, 5)) * np.nan
            for sgm in np.arange(nseg):
                for frm in np.arange(segments[0, sgm], segments[1, sgm] + 1):
                    arr[frm] = np.array(struct.unpack("ffffi", fid.read(20)))
            tracks[label] = arr.astype(np.float32)

    elif block["Format"] in [2]:  # by frame
        # get the labels
        labels = []
        for _ in np.arange(ntracks):
            labels += [_get_label(fid.read(256))]

        # get the available data
        tracks = (np.ones((nframes, 5)) * np.nan).astype(np.float32)
        tracks = {i: tracks.copy() for i in labels}
        for frm in np.arange(nframes):
            for trk in labels:
                vals = np.array(struct.unpack("ffffi", fid.read(20)))
                tracks[trk][frm, :] = vals.astype(np.float32)

    else:  # errors
        msg = f"block['Format'] must be 1, 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # convert the tracks in a single pandas dataframe
    obj = {}
    for trk, dfr in tracks.items():
        idx = pd.Index(np.arange(dfr.shape[0]) / freq + time0, name="TIME [s]")
        obj[trk] = pd.DataFrame(dfr, index=idx)

    return obj


def _read_data_generic(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read generic data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 14)
    if len(block) == 0:
        return None
    ntracks, freq, time0, nframes = struct.unpack("iifi", fid.read(16))

    # channels data
    chn_map = np.array(struct.unpack("h" * ntracks, fid.read(2 * ntracks)))

    # get the data
    if block["Format"] in [1]:
        tracks = _read_tracks(fid, nframes, ntracks, 1, True)
    elif block["Format"] in [2]:
        tracks = _read_frames(fid, nframes, ntracks, 1, True)
    else:
        msg = f"block['Format'] must be 1, 2, but {block['Format']}"
        msg += " was found."
        raise ValueError(msg)

    # convert the tracks in a single pandas dataframe
    tracks = pd.DataFrame(tracks)
    idx = pd.Index(np.arange(tracks.shape[0]) / freq + time0, name="TIME [s]")
    tracks.index = idx
    col = pd.MultiIndex.from_product(tracks.columns.to_numpy(), ["-"])  # type: ignore
    tracks.columns = col
    return {
        "TRACKS": tracks,
        "CHANNELS": chn_map.astype(np.int16),
    }


def _read_calibration_generic(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read calibration data for general purpose

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    calibration_data: dict[str, np.ndarray[Any, np.dtype[np.float_]]]
        the available calibration data.
    """

    # get the block and the number of signals
    fid, block = _get_block(fid, blocks, 15)
    if len(block) == 0:
        return None
    sig_n = struct.unpack("i", fid.read(4))[0]
    fid.seek(4, 1)

    # ge tthe channels map
    sig_map = np.array(list(struct.unpack(f"{sig_n}h", fid.read(2 * sig_n))))

    # get the calibration data
    sig_cal = np.nan * np.ones((sig_n, 3))
    for i in np.arange(sig_n):
        sig_cal[i, 0] = struct.unpack("i", fid.read(4))[0]
        sig_cal[i, 1:] = struct.unpack("ff", fid.read(8))

    return {
        "CHANNELS": sig_map.astype(np.int16),
        "DEVICE_TYPE": sig_cal[:, 0].astype(np.int32),
        "M": sig_cal[:, 1].astype(np.float32),
        "Q": sig_cal[:, 2].astype(np.float32),
    }


def _read_events(
    fid: BufferedReader,
    blocks: list[dict[str, int]],
):
    """
    read generic data sequence from a tdf file stream.

    Parameters
    ----------
    fid : BufferedReader
        the file stream as returned by the _open_tdf function.

    blocks: list[dict[str, int]],
        the list of blocks as returned by the _open_tdf function.

    Returns
    -------
    data: dict[str, Any]
        the available data.

    time0: float
        the starting time of the event.
    """

    # get the generic data
    fid, block = _get_block(fid, blocks, 16)
    if len(block) == 0:
        return None
    nevents, time0 = struct.unpack("if", fid.read(8))

    # read the events
    events: dict[str, Any] = {}
    if block["Format"] in [1]:
        for _ in np.arange(nevents):
            lbl = _get_label(fid.read(256))
            typ, nit = struct.unpack("ii", fid.read(8))
            data = struct.unpack("f" * nit, fid.read(nit * 4))
            events[lbl] = {"TYPE": typ, "DATA": data, "TIME0": time0}

    return events


def read_tdf(path: str):
    """
    Return the readings from a .tdf file.

    Parameters
    ----------
    path: str
        an existing tdf path.

    strip: bool (default = True)
        if True, tracked outcomes are automatically resized by removing
        complete missing outcomes from the ends

    Returns
    -------
    tdf: dict[str, Any]
        a dict containing the distinct data properly arranged by type.
    """

    # check the validity of the entered path
    assert_file_extension(path, "tdf")

    tdf_signature = "41604B82CA8411D3ACB60060080C6816"
    tdf: dict[str, dict[Any, Any] | None] = {}
    version = float("nan")

    # try opening the file
    fid = open(path, "rb")
    try:
        # check the signature
        blocks = []
        next_entry_offset = 40
        sig = struct.unpack("IIII", fid.read(16))
        sig = "".join([f"{b:08x}" for b in sig])
        if sig.upper() != tdf_signature:
            raise IOError("invalid file")

        # get the number of entries
        version, n_entries = struct.unpack("Ii", fid.read(8))
        assert n_entries > 0, "The file specified contains no data."

        # check each entry to find the available blocks
        for _ in range(n_entries):
            if -1 == fid.seek(next_entry_offset, 1):
                raise IOError("Error: the file specified is corrupted.")

            # get the data types
            block_info = struct.unpack("IIii", fid.read(16))
            if block_info[1] != 0:  # Format != 0 ensures valid blocks
                blocks += [dict(zip(_BLOCK_KEYS, block_info))]  # type: ignore

            # update the offset
            next_entry_offset = 272

        # read all entries
        tdf["CAMERA"] = {
            "TRACKED": _read_camera_tracked(fid, blocks),
            "RAW": _read_camera_raw(fid, blocks),
            "PARAMETERS": _read_camera_params(fid, blocks),
            "CALIBRATION": _read_camera_calibration(fid, blocks),
            "CONFIGURATION": _read_camera_configuration(fid, blocks),
        }
        tdf["FORCE_PLATFORM"] = {
            "TRACKED": _read_platforms_tracked(fid, blocks),
            "RAW": _read_platforms_raw(fid, blocks),
            "PARAMETERS": _read_platforms_params(fid, blocks),
            "CALIBRATION": _read_platforms_calibration(fid, blocks),
        }
        tdf["GENERIC"] = {
            "DATA": _read_data_generic(fid, blocks),
            "CALIBRATION": _read_calibration_generic(fid, blocks),
        }
        tdf["EMG"] = _read_emg(fid, blocks)
        tdf["EVENTS"] = _read_events(fid, blocks)
        tdf["VOLUME"] = _read_volume(fid, blocks)
        tdf["VERSION"] = version

    except Exception as exc:
        raise RuntimeError(exc) from exc

    finally:
        fid.close()

    return tdf
