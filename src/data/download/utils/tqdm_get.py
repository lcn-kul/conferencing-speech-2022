from pathlib import Path
import requests
import time
from tqdm import tqdm


from src.data.download.utils.tqdm_write import tqdm_print


def tqdm_get(
    url: str,
    dest_path: Path,
    name: str,
    idx: int = 0,
):
    # Streaming, so we can iterate over the response.
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    if total_size_in_bytes == 0:
        tqdm_print("Content length unknown...", name, idx)
        time.sleep(2)
    block_size = 1024  # 1 Kibibyte
    with tqdm(
        desc=name,
        total=total_size_in_bytes,
        unit='iB',
        unit_scale=True,
        position=idx,
        leave=False
    ) as pbar:
        with open(dest_path, 'wb') as file:
            for data in response.iter_content(block_size):
                pbar.update(len(data))
                file.write(data)
