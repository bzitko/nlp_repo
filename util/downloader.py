import sys
IS_GOOGLE_COLAB = "google.colab" in sys.modules
import os.path
import urllib.request
import urllib.parse


URL = "https://raw.githubusercontent.com/bzitko/nlp_repo/main"

def download(url, filename):
        print(f"Downloading {filename} ", end="")
        urllib.request.urlretrieve(url, filename)
        print("DONE!")


def prepare(relative_url):

    download_parts = relative_url.split("/")
    download_filename = download_parts[-1]
    filename_parts = download_filename.split(".")
    filename_ext = filename_parts[-1]

    is_compressed = filename_ext in ("bz2", "zip")
    if is_compressed:
        filename = ".".join(filename_parts[:-1])
    else:
        filename = ".".join(filename_parts)

    # check if final filename
    if os.path.exists(filename):
        return
    
    url = "/".join([URL] + [urllib.parse.quote(part) for part in download_parts])
    
    if is_compressed :
        if not os.path.exists(download_filename):
            download(url, download_filename)
        uncompress(download_filename, filename)
    else:
        download(url, filename)
    
def uncompress(compressed_filename, uncompressed_filename):
    compressed_parts = compressed_filename.split(".")
    ext = compressed_parts[-1]
    print(f"Uncompressing to {uncompressed_filename} ", end="")
    if ext == "bz2":
        import bz2
        with open(compressed_filename, 'rb') as bzf, open(uncompressed_filename, 'wb') as fp:
            fp.write(bz2.decompress(bzf.read()))    
    elif ext == "zip":
        from zipfile import ZipFile
        with ZipFile(compressed_filename) as zf:
            zf.extract(uncompressed_filename)

    if IS_GOOGLE_COLAB:
        os.remove(compressed_filename)
    print("DONE!")     
