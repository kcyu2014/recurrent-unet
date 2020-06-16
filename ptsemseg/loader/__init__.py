import json

import yaml

from ptsemseg.loader.gtea_hand_loader import gteaHandLoader
from ptsemseg.loader.handoverface_hand_loader import handoverfaceHandLoader
from ptsemseg.loader.eyth_hand_loader import eythHandLoader
from ptsemseg.loader.ego_hand_loader import egoHandLoader
from ptsemseg.loader.epfl_hand_loader import epflHandLoader, EPFLROILoader
from ptsemseg.loader.pascal_voc_loader import pascalVOCLoader
from ptsemseg.loader.camvid_loader import camvidLoader
from ptsemseg.loader.ade20k_loader import ADE20KLoader
from ptsemseg.loader.mit_sceneparsing_benchmark_loader import (
    MITSceneParsingBenchmarkLoader
)
from ptsemseg.loader.cityscapes_loader import cityscapesLoader
from ptsemseg.loader.nyuv2_loader import NYUv2Loader
from ptsemseg.loader.road_loader import RoadLoader
from ptsemseg.loader.sunrgbd_loader import SUNRGBDLoader
from ptsemseg.loader.mapillary_vistas_loader import mapillaryVistasLoader
from ptsemseg.loader.drive_loader import driveLoader


def get_loader(name, roi_only=False):
    """get_loader

    :param name:
    """

    return {
        "pascal": pascalVOCLoader,
        "camvid": camvidLoader,
        "ade20k": ADE20KLoader,
        "mit_sceneparsing_benchmark": MITSceneParsingBenchmarkLoader,
        "cityscapes": cityscapesLoader,
        "road": RoadLoader,
        "nyuv2": NYUv2Loader,
        "sunrgbd": SUNRGBDLoader,
        "vistas": mapillaryVistasLoader,
        "epfl_hand": epflHandLoader,
        "epfl_hand_roi": EPFLROILoader,
        "ego_hand": egoHandLoader,
        "eyth_hand": eythHandLoader,
        "gtea_hand": gteaHandLoader,
        "hof_hand": handoverfaceHandLoader,
        "drive": driveLoader,
    }[name]


def get_data_path(name, config_file="config.json"):
    """get_data_path

    :param name:
    :param config_file:
    """
    try:
        data = json.load(open(config_file))
        path = data[name]["data_path"]
    except Exception as e:
        print(e)
        data = yaml.load(open(config_file))
        path = data['data']['path']
    return path


def get_void_class(name):
    if name in ['epfl_hand_roi', 'road']:
        return True
    return False