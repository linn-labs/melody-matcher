import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# API credentials
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")
LASTFM_API_SECRET = os.getenv("LASTFM_API_SECRET")
LASTFM_API_BASE = "https://ws.audioscrobbler.com/2.0/"

# Rate limiting
API_RATE_LIMIT = 4.5  # requests per second (10% safety margin on 5/sec)
SCRAPE_RATE_LIMIT = 1.0  # requests per second for web scraping
API_MAX_RETRIES = 3
API_BASE_TIMEOUT = 10  # seconds

# Database
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "melody_matcher.db"

# Collection thresholds
MIN_SCROBBLES = 500  # minimum total scrobbles to consider a user
MIN_UNIQUE_TRACKS = 100  # minimum unique tracks for a useful history
MAX_TOP_TRACKS_PAGES = 5  # up to 1,000 tracks per user
TOP_TRACKS_PER_PAGE = 200
SCRAPE_PAGES_PER_ARTIST = 9  # ~250 users per artist
FRIENDS_SAMPLE_SIZE = 200  # users to sample for friends expansion
FRIENDS_PER_USER = 50  # friends to fetch per user

# Diversity scoring thresholds
MIN_SUPER_GENRES = 3  # minimum distinct super-genres
MAX_TOP5_CONCENTRATION = 0.60  # top 5 artists must be < 60% of plays

# Curated seed artists spanning 12+ genres for scraping diversity
# Goal: discover users with diverse taste by starting from many genre entry points
SEED_ARTISTS = [
    # Pop
    "Taylor Swift", "Billie Eilish", "Charli XCX", "Lorde", "Kate Bush",
    # Rock
    "Radiohead", "Arctic Monkeys", "Queens of the Stone Age", "The Strokes",
    "King Gizzard & The Lizard Wizard",
    # Hip-Hop / Rap
    "Kendrick Lamar", "MF DOOM", "Tyler, the Creator", "OutKast", "Little Simz",
    # Electronic / Dance
    "Aphex Twin", "Burial", "Four Tet", "Boards of Canada", "Sophie",
    # Commercial Dance / EDM (gap fill)
    "Claptone", "Louis The Child", "Gryffin", "Robin Schulz",
    "MEDUZA", "Tinlicker", "Kygo",
    # Metal
    "Opeth", "Deafheaven", "Mastodon", "Between the Buried and Me", "Gojira",
    # Indie / Alternative
    "Sufjan Stevens", "Björk", "Animal Collective", "LCD Soundsystem",
    "Tame Impala",
    # Classical / Neoclassical
    "Johann Sebastian Bach", "Claude Debussy", "Steve Reich",
    "Nils Frahm", "Max Richter",
    # Jazz
    "Miles Davis", "John Coltrane", "Kamasi Washington", "Robert Glasper",
    "Hiromi",
    # Country / Folk / Americana
    "Sturgill Simpson", "Jason Isbell", "Townes Van Zandt",
    "Gillian Welch", "Tyler Childers",
    # Texas / Red Dirt Country (gap fill)
    "Mike Ryan", "Cody Johnson", "Shotgun Rider", "Wade Bowen",
    "Randy Rogers Band", "Parker McCollum", "Lainey Wilson",
    "Triston Marez", "HARDY", "Florida Georgia Line",
    "Luke Combs", "Maren Morris", "Scotty McCreery",
    # R&B / Soul
    "Frank Ocean", "Erykah Badu", "D'Angelo", "Solange", "SZA",
    # Latin
    "Rosalía", "Bad Bunny", "Natalia Lafourcade", "Bomba Estéreo",
    # World / Global
    "Fela Kuti", "Tinariwen", "Khruangbin", "Mdou Moctar", "Mulatu Astatke",
]

# Super-genre mapping: keyword patterns → super-genre label
# Used for classifying Last.fm artist tags into broad categories
SUPER_GENRE_MAP = {
    "pop": ["pop", "synth pop", "synthpop", "dream pop", "indie pop",
            "art pop", "chamber pop", "electropop", "k-pop"],
    "rock": ["rock", "punk", "grunge", "shoegaze", "post-rock", "post-punk",
             "garage rock", "psychedelic rock", "progressive rock", "stoner rock",
             "noise rock", "math rock", "emo", "hardcore"],
    "hip-hop": ["hip-hop", "hip hop", "rap", "trap", "grime", "boom bap",
                "conscious hip hop", "underground hip hop"],
    "electronic": ["electronic", "techno", "house", "ambient", "idm",
                   "drum and bass", "dnb", "dubstep", "trance", "edm",
                   "glitch", "breakbeat", "downtempo", "chillwave",
                   "synthwave", "vaporwave"],
    "metal": ["metal", "death metal", "black metal", "doom metal",
              "progressive metal", "thrash metal", "metalcore",
              "post-metal", "sludge metal", "heavy metal"],
    "indie": ["indie", "indie rock", "lo-fi", "alternative",
              "alternative rock", "slowcore", "midwest emo"],
    "classical": ["classical", "contemporary classical", "baroque",
                  "romantic", "opera", "orchestral", "chamber music",
                  "neoclassical", "minimalism", "modern classical"],
    "jazz": ["jazz", "free jazz", "fusion", "bebop", "cool jazz",
             "acid jazz", "nu jazz", "jazz fusion", "smooth jazz"],
    "country": ["country", "folk", "americana", "bluegrass",
                "country rock", "alt-country", "singer-songwriter",
                "folk rock", "freak folk"],
    "r&b": ["r&b", "rnb", "soul", "neo-soul", "funk", "motown",
            "rhythm and blues", "neo soul"],
    "latin": ["latin", "reggaeton", "bossa nova", "salsa", "cumbia",
              "latin pop", "latin rock", "bachata", "merengue"],
    "world": ["world", "afrobeat", "afropop", "desert blues",
              "ethio-jazz", "world music", "african", "tuareg",
              "highlife", "dub", "reggae", "ska"],
}

# Scrobble collection
SCROBBLES_PER_PAGE = 200
MAX_SCROBBLE_PAGES = 0         # 0 = collect all pages (no limit)
SCROBBLE_BATCH_COMMIT = 2000   # scrobble records per commit

# Deezer
DEEZER_API_BASE = "https://api.deezer.com"
DEEZER_RATE_LIMIT = 10.0  # requests per second
DEEZER_MAX_RETRIES = 5

# MERT
MERT_MODEL_ID = "m-a-p/MERT-v1-330M"
MERT_SAMPLE_RATE = 24000
MERT_EMBEDDING_DIM = 1024
MERT_BATCH_SIZE = 8  # 30-sec clips per batch on RTX 3060 (12GB VRAM, ~8GB used by model)

# Embeddings
EMBEDDINGS_PATH = DATA_DIR / "embeddings.h5"
EMBEDDINGS_NPY_PATH = DATA_DIR / "embeddings.npy"

# Training - data
TRAIN_CONTEXT_RATIO = 0.7
TRAIN_WINDOW_SIZE = 300
TRAIN_BATCH_SIZE = 32
TRAIN_NUM_WORKERS = int(os.getenv("TRAIN_NUM_WORKERS", "8"))
TRAIN_PREFETCH_FACTOR = int(os.getenv("TRAIN_PREFETCH_FACTOR", "4"))
TRAIN_VAL_TEST_SPLIT = (0.8, 0.1, 0.1)
MIN_LIBRARY_SIZE = 50
TRAINING_DATA_DIR = DATA_DIR / "training"

# Training - optimization
TRAIN_LEARNING_RATE = 1e-4
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_WARMUP_RATIO = 0.05
TRAIN_EPOCHS = 20
TRAIN_PATIENCE = 5
TRAIN_GRAD_CLIP = 1.0
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
