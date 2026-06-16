import os
import logging
import warnings

from minigpt4.common.registry import registry
from minigpt4.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from minigpt4.datasets.datasets.laion_dataset import LaionDataset
from minigpt4.datasets.datasets.cc_sbu_dataset import CCSBUDataset, CCSBUAlignDataset
from minigpt4.datasets.datasets.coco_vqa_datasets import COCOVQADataset
from minigpt4.datasets.datasets.coco_caption import COCOCapDataset
from minigpt4.datasets.datasets.aok_vqa_datasets import AOKVQADataset
from minigpt4.datasets.datasets.coco_vqa_datasets import PrivacyPreferenceOptimization, PrivacyGradientAscent
from minigpt4.datasets.datasets.coco_vqa_datasets import SafeErase_Harmimg_Unharmtxt, SafeErase_Unharmimg_Harmtxt
from minigpt4.datasets.datasets.coco_vqa_datasets import ImageNetR_unlearn_GA, ImageNetR_unlearn_PO


@registry.register_builder("cc_sbu")
class CCSBUBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/cc_sbu/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets


@registry.register_builder("laion")
class LaionBuilder(BaseDatasetBuilder):
    train_dataset_cls = LaionDataset

    DATASET_CONFIG_DICT = {"default": "configs/datasets/laion/defaults.yaml"}

    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()

        build_info = self.config.build_info

        datasets = dict()
        split = "train"

        # create datasets
        # [NOTE] return inner_datasets (wds.DataPipeline)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            location=build_info.storage,
        ).inner_dataset

        return datasets

###################################################
##################   Captioning   #################
###################################################

@registry.register_builder("coco_caption") # FLickr
class COCOCapBuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOCapDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/caption.yaml",
    }
    
    def build_datasets(self, task_id):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.images.storage
        caption_path = '/workspace/datasets/Flicker30K/Flickr30K-factory/splitted_final'
        taskid2anno = {'0': 'F30K_animals_train.json', 
                       '1': 'F30K_instruments_train.json', 
                       '2': 'F30K_scene_train.json', 
                       '3': 'F30K_vehicles_train.json'}
        cap_name = taskid2anno[str(task_id)]
        
        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        
        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            # ann_paths=[os.path.join(storage_path, 'filter_cap_first50_task5.json')],
            ann_paths=[os.path.join(caption_path, cap_name)],
            vis_root=os.path.join(storage_path, 'train_image'),
        )
        return datasets
    
###################################################
#################       VQA       #################
###################################################

@registry.register_builder("coco_vqa") # COCO 
class COCOVQABuilder(BaseDatasetBuilder):
    train_dataset_cls = COCOVQADataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/coco/coco_qa.yaml",
    }
    
    def build_datasets(self, task_id):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.images.storage
        caption_path = '/workspace/datasets/MS-COCO14/annotations/COCO-QA/types_train/'

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        cap_name = str(task_id) + '.json'

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            # ann_paths=[os.path.join(storage_path, 'filter_cap_first50_task5.json')],
            ann_paths=[os.path.join(caption_path, cap_name)],
            vis_root=os.path.join(storage_path),
        )
        return datasets

###################################################
################## Classification #################
###################################################
@registry.register_builder("cc_sbu_align")
class CCSBUAlignBuilder(BaseDatasetBuilder):
    train_dataset_cls = CCSBUAlignDataset

    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/cc_sbu/align.yaml",
    }

    def build_datasets(self, task_id):
        # at this point, all the annotations and image/videos should be all downloaded to the specified locations.
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.storage

        datasets = dict()

        if not os.path.exists(storage_path):
            warnings.warn("storage path {} does not exist.".format(storage_path))

        # create datasets
        dataset_cls = self.train_dataset_cls
        cap_name = '10_task_order2/train/task' + str(task_id) + '.json'

        datasets['train'] = dataset_cls(
            vis_processor=self.vis_processors["train"],
            text_processor=self.text_processors["train"],
            # ann_paths=[os.path.join(storage_path, 'filter_cap_first50_task5.json')],
            ann_paths=[os.path.join(storage_path, cap_name)],
            vis_root=os.path.join(storage_path, 'train_image'),
        )

        return datasets


# keywords = {
#     'Violence': ['FamilyAbuse', 'PhysicalAssault', 'TerroristAttacks', 'DestroyBridges', 'BombAttack', 'DerailTrains', 'Self-Harm', 'MilitantInsurgencies', 'Robbery', 'CarryingWeapons'], 
#     'IllegalActivity': ['Human Rights Violations', 'Illegal Firearms', 'Purchase Illegal Drugs', 'Counterfeit Currency', 'Illegal Immigration', 'Drunk Driving', 'Counterfeiting Fashion Items', 'Wildlife Trafficking', 'Hack Into Bank Account', 'Organ Harvesting'], 
#     'Weapon': ['Tank', 'Gun', 'WeaponizedRobot', 'NuclearWeaponry', 'Grenades', 'MilitaryAircraft', 'Torpedo', 'ImprovisedExplosiveDevice', 'ChemicalWeapon', 'BiologicalWeapon'], 
#     'Sex': ['porn games', 'hentai', 'sexuality', 'masturbate', 'porn streaming media', 'erotica', 'sexy toys', 'nude', 'bdsm', 'bestiality'], 
#     'Privacy': ['peeking others', 'over the shoulder peeking screen', 'privacy intruding camera', 'unauthorized access Personal Identification', 'secretly record', 'unauthorized access to chat history', 'mail stealing', 'drone surveillance', 'unauthorized access privacy photos', 'stalke someone'], 
#     'HateSpeech': ['Jewish people', 'Black people', 'Lesbian', 'Cyberbullying', 'Gay', 'Disabled individuals', 'Elderly people', 'Women', 'Communism', 'Obese people']
# }


keywords = [
            'porn games', 'hentai', 'sexuality', 'masturbate', 'porn streaming media', 'erotica', 'sexy toys', 'nude', 'bdsm', 'bestiality', 
            'FamilyAbuse', 'PhysicalAssault', 'TerroristAttacks', 'DestroyBridges', 'BombAttack', 'DerailTrains', 'Self-Harm', 'MilitantInsurgencies', 'Robbery', 'CarryingWeapons', 
            'Human Rights Violations', 'Illegal Firearms', 'Purchase Illegal Drugs', 'Counterfeit Currency', 'Illegal Immigration', 'Drunk Driving', 'Counterfeiting Fashion Items', 'Wildlife Trafficking', 'Hack Into Bank Account', 'Organ Harvesting', 
            'Tank', 'Gun', 'WeaponizedRobot', 'NuclearWeaponry', 'Grenades', 'MilitaryAircraft', 'Torpedo', 'ImprovisedExplosiveDevice', 'ChemicalWeapon', 'BiologicalWeapon', 
            
            'peeking others', 'over the shoulder peeking screen', 'privacy intruding camera', 'unauthorized access Personal Identification', 'secretly record', 'unauthorized access to chat history', 'mail stealing', 'drone surveillance', 'unauthorized access privacy photos', 'stalke someone', 
            'Jewish people', 'Black people', 'Lesbian', 'Cyberbullying', 'Gay', 'Disabled individuals', 'Elderly people', 'Women', 'Communism', 'Obese people']

def split_keywords(keywords, split_sizes):
    """
    keywords 리스트를 split_sizes 패턴에 따라 반복적으로 잘라 반환
    예: split_sizes = [3, 4, 3]
    """
    result, idx = [], 0
    n = len(keywords)
    while idx < n:
        for size in split_sizes:
            if idx >= n:
                break
            result.append(keywords[idx: idx + size])
            idx += size
    return result

# Base class to avoid redundancy
class PrivacyBaseBuilder(BaseDatasetBuilder):
    TRAIN_ANNO = '/workspace/datasets/safe_eraser/dataset/all_train.json'
    TASK_to_KEYWORDS = split_keywords(keywords, [5])
    DATASET_CONFIG_DICT = {
    }
    
    def build_datasets(self, task_id):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.images.storage

        if not os.path.exists(storage_path):
            warnings.warn(f"Storage path {storage_path} does not exist.")
        
        print(f'[DATA] Total Tasks : {len(self.TASK_to_KEYWORDS)} / Current task id: {task_id}')
        print(f'[DATA] {len(self.TASK_to_KEYWORDS[int(task_id)])} keywords')
        ann_path = self.TRAIN_ANNO
        datasets = dict()

        datasets['train'] =  self.train_dataset_cls(
                        vis_processor=self.vis_processors["train"],
                        text_processor=self.text_processors["train"],
                        ann_paths=[ann_path],
                        vis_root=storage_path,
                        keywords = self.TASK_to_KEYWORDS[int(task_id)]
                        )
        
        return datasets
    
@registry.register_builder("privacy_preference_optimization")
class PrivacyPreferenceOptimizationBuilder(PrivacyBaseBuilder):
    train_dataset_cls = PrivacyPreferenceOptimization
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/safe_erase/safe_erase_PO.yaml",
    }


@registry.register_builder("privacy_gradient_ascent")
class PrivacyGradientAscentBuilder(PrivacyBaseBuilder):
    train_dataset_cls = PrivacyGradientAscent
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/safe_erase/safe_erase_GA.yaml",
    }

class Safety_Harmimg_Unharmtxt(PrivacyBaseBuilder):
    EVAL_ANNO = '/workspace/datasets/safe_eraser/dataset/all_val.json'
    TASK_to_KEYWORDS = split_keywords(keywords, [5])
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/safe_erase/safe_erase_PO.yaml",
    }

    train_dataset_cls = SafeErase_Harmimg_Unharmtxt

class Safety_Unharmimg_Harmtxt(PrivacyBaseBuilder):
    EVAL_ANNO = '/workspace/datasets/safe_eraser/dataset/all_val.json'
    TASK_to_KEYWORDS = split_keywords(keywords, [5])
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/safe_erase/safe_erase_PO.yaml",
    }

    train_dataset_cls = SafeErase_Unharmimg_Harmtxt


imagenet_r_keywords = [['bison', 'black_swan', 'chow_chow', 'violin', 'lawn_mower', 'lion', 'broom', 'badger', 'hammer', 'skunk', 
                        'afghan_hound', 'military_aircraft', 'mobile_phone', 'spider_web', 'duck', 'fly', 'fox_squirrel', 'cockroach', 'pineapple', 'saint_bernard',], 
 
                        [ 'boxer', 'rugby_ball', 'burrito', 'banana', 'acorn', 'toucan', 'tarantula', 'dalmatian', 'birdhouse', 'jeep', 
                        'lipstick',  'gorilla', 'hotdog', 'shield', 'ant', 'cauldron', 'mailbox', 'guillotine', 'lobster', 'hen', ], 
                        
                        [ 'bathtub', 'revolver', 'ladybug', 'timber_wolf', 'scorpion', 'strawberry', 'gibbon', 'lighthouse', 'stingray', 'whippet', 
                        'border_collie', 'ice_cream', 'steam_locomotive', 'sandal', 'basset_hound', 'tank', 'polar_bear', 'cucumber', 'tiger', 'harmonica', ], 
                        
                        [ 'grand_piano', 'electric_guitar', 'german_shepherd_dog', 'candle', 'yorkshire_terrier', 'baboon', 'bucket', 'backpack', 'pirate_ship', 'harp', 
                        'leopard', 'husky', 'trombone', 'goose', 'bow_tie', 'accordion', 'west_highland_white_terrier', 'collie', 'pelican','hippopotamus']]


# imagenet_r_keywords = [['bison', 'black_swan', 'chow_chow', 'violin', 'lawn_mower', 'lion', 'broom', 'badger', 'hammer', 'skunk', 
#                         'afghan_hound', 'military_aircraft', 'mobile_phone', 'spider_web', 'duck', 'fly', 'fox_squirrel', 'cockroach', 'pineapple', 'saint_bernard',
 
#                         'boxer', 'rugby_ball', 'burrito', 'banana', 'acorn', 'toucan', 'tarantula', 'dalmatian', 'birdhouse', 'jeep', 
#                         'lipstick',  'gorilla', 'hotdog', 'shield', 'ant', 'cauldron', 'mailbox', 'guillotine', 'lobster', 'hen',
                        
#                         'bathtub', 'revolver', 'ladybug', 'timber_wolf', 'scorpion', 'strawberry', 'gibbon', 'lighthouse', 'stingray', 'whippet', 
#                         'border_collie', 'ice_cream', 'steam_locomotive', 'sandal', 'basset_hound', 'tank', 'polar_bear', 'cucumber', 'tiger', 'harmonica', 
                        
#                         'grand_piano', 'electric_guitar', 'german_shepherd_dog', 'candle', 'yorkshire_terrier', 'baboon', 'bucket', 'backpack', 'pirate_ship', 'harp', 
#                         'leopard', 'husky', 'trombone', 'goose', 'bow_tie', 'accordion', 'west_highland_white_terrier', 'collie', 'pelican','hippopotamus']]

# Base class to avoid redundancy
class PrivacyBaseBuilder_ImageNetR(BaseDatasetBuilder):
    TRAIN_ANNO = '/workspace/datasets/imagenet-r/imagenet_r_train.json'
    TASK_to_KEYWORDS = imagenet_r_keywords
    DATASET_CONFIG_DICT = {
    }
    
    def build_datasets(self, task_id):
        logging.info("Building datasets...")
        self.build_processors()

        build_info = self.config.build_info
        storage_path = build_info.images.storage

        if not os.path.exists(storage_path):
            warnings.warn(f"Storage path {storage_path} does not exist.")
        
        print(f'[DATA] Total Tasks : {len(self.TASK_to_KEYWORDS)} / Current task id: {task_id}')
        print(f'[DATA] {len(self.TASK_to_KEYWORDS[int(task_id)])} keywords')
        ann_path = self.TRAIN_ANNO
        datasets = dict()

        datasets['train'] =  self.train_dataset_cls(
                        vis_processor=self.vis_processors["train"],
                        text_processor=self.text_processors["train"],
                        ann_paths=[ann_path],
                        vis_root=storage_path,
                        keywords = self.TASK_to_KEYWORDS[int(task_id)]
                        )
        
        return datasets
    
@registry.register_builder("privacy_gradient_ascent_imagenet_r")
class Privacy_gradient_ascent_imagenet_r(PrivacyBaseBuilder_ImageNetR):
    train_dataset_cls = ImageNetR_unlearn_GA
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/safe_erase/safe_erase_GA_imagenet_r.yaml",
    }

@registry.register_builder("privacy_preference_optimization_imagenet_r")
class Privacy_preference_optimization_imagenet_r(PrivacyBaseBuilder_ImageNetR):
    train_dataset_cls = ImageNetR_unlearn_PO
    DATASET_CONFIG_DICT = {
        "default": "configs/datasets/safe_erase/safe_erase_PO_imagenet_r.yaml",
    }