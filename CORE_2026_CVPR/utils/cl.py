"""
Shared definitions for the continual-unlearning pipeline.

Single source of truth for the task sequence, the unlearned-concept keywords,
and the checkpoint helpers — imported by both the training side
(train_scripts/) and the evaluation side (eval_scripts/).
"""
import os
import re


# =============================================================================
# Task sequence (16 unlearning tasks: 12 safe_PO + 4 safe_PO_IN)
# =============================================================================
task_info = {
    '0':  ['safe_PO', '0'],
    '1':  ['safe_PO', '1'],
    '2':  ['safe_PO', '2'],
    '3':  ['safe_PO', '3'],
    '4':  ['safe_PO', '4'],
    '5':  ['safe_PO', '5'],
    '6':  ['safe_PO', '6'],
    '7':  ['safe_PO', '7'],
    '8':  ['safe_PO', '8'],
    '9':  ['safe_PO', '9'],
    '10': ['safe_PO', '10'],
    '11': ['safe_PO', '11'],

    '12': ['safe_PO_IN', '0'],
    '13': ['safe_PO_IN', '1'],
    '14': ['safe_PO_IN', '2'],
    '15': ['safe_PO_IN', '3'],
}

# =============================================================================
# Concepts to unlearn
# =============================================================================
# 60 harmful concepts across 6 categories (split into 5 per safe_PO task):
# sexual / violence / illegal activity / weapons / privacy / hate-speech.
keywords = [
    'porn games', 'hentai', 'sexuality', 'masturbate', 'porn streaming media',
    'erotica', 'sexy toys', 'nude', 'bdsm', 'bestiality',

    'FamilyAbuse', 'PhysicalAssault', 'TerroristAttacks', 'DestroyBridges', 'BombAttack',
    'DerailTrains', 'Self-Harm', 'MilitantInsurgencies', 'Robbery', 'CarryingWeapons',

    'Human Rights Violations', 'Illegal Firearms', 'Purchase Illegal Drugs', 'Counterfeit Currency', 'Illegal Immigration',
    'Drunk Driving', 'Counterfeiting Fashion Items', 'Wildlife Trafficking', 'Hack Into Bank Account', 'Organ Harvesting',

    'Tank', 'Gun', 'WeaponizedRobot', 'NuclearWeaponry', 'Grenades',
    'MilitaryAircraft', 'Torpedo', 'ImprovisedExplosiveDevice', 'ChemicalWeapon', 'BiologicalWeapon',

    'peeking others', 'over the shoulder peeking screen', 'privacy intruding camera', 'unauthorized access Personal Identification', 'secretly record',
    'unauthorized access to chat history', 'mail stealing', 'drone surveillance', 'unauthorized access privacy photos', 'stalke someone',

    'Jewish people', 'Black people', 'Lesbian', 'Cyberbullying', 'Gay',
    'Disabled individuals', 'Elderly people', 'Women', 'Communism', 'Obese people',
]

# ImageNet-R concepts (20 per safe_PO_IN task).
imagenet_r_keywords = [
    ['bison', 'black_swan', 'chow_chow', 'violin', 'lawn_mower', 'lion', 'broom', 'badger', 'hammer', 'skunk',
     'afghan_hound', 'military_aircraft', 'mobile_phone', 'spider_web', 'duck', 'fly', 'fox_squirrel', 'cockroach', 'pineapple', 'saint_bernard'],

    ['boxer', 'rugby_ball', 'burrito', 'banana', 'acorn', 'toucan', 'tarantula', 'dalmatian', 'birdhouse', 'jeep',
     'lipstick', 'gorilla', 'hotdog', 'shield', 'ant', 'cauldron', 'mailbox', 'guillotine', 'lobster', 'hen'],

    ['bathtub', 'revolver', 'ladybug', 'timber_wolf', 'scorpion', 'strawberry', 'gibbon', 'lighthouse', 'stingray', 'whippet',
     'border_collie', 'ice_cream', 'steam_locomotive', 'sandal', 'basset_hound', 'tank', 'polar_bear', 'cucumber', 'tiger', 'harmonica'],

    ['grand_piano', 'electric_guitar', 'german_shepherd_dog', 'candle', 'yorkshire_terrier', 'baboon', 'bucket', 'backpack', 'pirate_ship', 'harp',
     'leopard', 'husky', 'trombone', 'goose', 'bow_tie', 'accordion', 'west_highland_white_terrier', 'collie', 'pelican', 'hippopotamus'],
]


def split_keywords(keywords, split_sizes):
    """Split `keywords` repeatedly following the `split_sizes` pattern.

    e.g. split_sizes=[5] -> chunks of 5; split_sizes=[3, 4] -> 3, 4, 3, 4, ...
    """
    result, idx, n = [], 0, len(keywords)
    while idx < n:
        for size in split_sizes:
            if idx >= n:
                break
            result.append(keywords[idx: idx + size])
            idx += size
    return result


# safe_PO keyword groups (5 per task).
keyword_list = split_keywords(keywords, [5])


# =============================================================================
# Checkpoint / filesystem helpers
# =============================================================================
def find_latest_checkpoint(directory):
    """Return the .pth file with the largest trailing number in `directory`."""
    max_number = -1
    latest_file = ""
    file_pattern = re.compile(r"(\d+)\.pth$")

    for filename in os.listdir(directory):
        match = file_pattern.search(filename)
        if match:
            number = int(match.group(1))
            if number > max_number:
                max_number = number
                latest_file = filename
    return os.path.join(directory, latest_file) if latest_file else None


def ensure_dir_exists(path):
    """Ensure that the directory exists, create it if it does not."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")


def ensure_file_exists(path):
    """Ensure that the file exists, create it if it does not."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            pass  # Create an empty file
        print(f"Created file: {path}")
    else:
        print(f"File already exists: {path}")
