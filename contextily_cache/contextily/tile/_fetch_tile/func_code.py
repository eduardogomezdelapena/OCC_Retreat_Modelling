# first line: 312
def _fetch_tile(tile_url, wait, max_retries):
    array = _retryer(tile_url, wait, max_retries)
    return array
