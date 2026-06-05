from Dassl.dassl.utils import Registry, check_availability
from datasets.caltech101 import Caltech101
from datasets.oxford_flowers import OxfordFlowers
from datasets.oxford_pets import OxfordPets
from datasets.food101 import Food101
from datasets.dtd import DescribableTextures
from datasets.ucf101 import UCF101
from datasets.stanford_cars import StanfordCars
from datasets.fgvc_aircraft import FGVCAircraft
from datasets.eurosat import EuroSAT

DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.register(Caltech101)
DATASET_REGISTRY.register(OxfordFlowers)
DATASET_REGISTRY.register(OxfordPets) 
DATASET_REGISTRY.register(Food101)
DATASET_REGISTRY.register(DescribableTextures)
DATASET_REGISTRY.register(UCF101)
DATASET_REGISTRY.register(StanfordCars)
DATASET_REGISTRY.register(FGVCAircraft)
DATASET_REGISTRY.register(EuroSAT)


def build_dataset(cfg):
    avai_datasets = DATASET_REGISTRY.registered_names()
    check_availability(cfg.DATASET.NAME, avai_datasets)
    if cfg.VERBOSE:
        print("Loading dataset: {}".format(cfg.DATASET.NAME))
    return DATASET_REGISTRY.get(cfg.DATASET.NAME)(cfg)
