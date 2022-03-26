from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
import io
import json
import multiprocessing as mp
from pathlib import Path
import sys
from time import sleep
import traceback
# from typing import TypedDict # Not in Python3.6?


# ============== #
# HELPER METHODS #
# ============== #

# Constructing queries.
# Source:
# 1: https://developers.google.com/drive/api/v3/search-files
# 2: https://stackoverflow.com/a/69726030


def _q_files_in_folder(folder_id: str):
    return ' and '.join((
        "'%s' in parents" % folder_id,
        "mimeType != 'application/vnd.google-apps.folder'"
    ))


def _q_folders_in_folder(folder_id: str):
    return " and ".join((
        "'%s' in parents" % folder_id,
        "mimeType = 'application/vnd.google-apps.folder'"
    ))

# Printing progress bar...


def _progress_bar(idx: int, N: int, bar_len: int):
    progress = (idx+1) / N
    max_bars = bar_len - 2
    cur_bars = round(progress * bar_len)
    cur_spaces = max_bars - cur_bars
    bar = "[" + "="*cur_bars + " "*cur_spaces + "]"
    bar += " (%0.2f %%)" % (progress*100)
    return bar


def _print_progress(idx: int, N: int):
    bar_len = 70
    bar = _progress_bar(idx, N, bar_len)
    print(bar, end="\r")

# Clearing line in terminal....


def _clear_line():
    sys.stdout.write("\033[K")

# ============ #
# CORE CLASSES #
# ============ #


# class GoogleDriveItem(TypedDict):
#     id: str
#     name: str


class GoogleDriveDownloader():
    def __init__(self, credential_path: Path):
        """The GoogleDriveDownloader can be used to download files and folders
        from Google Drive.

        # Requirements:

        1. Create Google Cloud project and Service Key.
           https://cloud.google.com/docs/authentication/getting-started#creating_a_service_account
           By following these instructions, you will download a JSON file
           containing your Google credentials. Place this JSON file in the
           following location:
           ```
           conferencing-speech-2022/gdrive_creds.json
           ```
        2. Enable Google Drive API for this project.
           https://console.developers.google.com/apis/library/drive.googleapis.com
        3. Wait 5 minutes for changes to propagate through Google systems.


        Args:
            credential_path (Path): Path to Google Drive credentials JSON file.
        """
        self.credential_path = credential_path

        # Load credentials.
        if not self.credential_path.exists():
            msg = "Missing gdrive_creds.json. Please read the main "
            msg += "README to learn how to create it."
            print(msg)
            exit(1)
        with open(self.credential_path) as f:
            creds_json = json.load(f)

        self.credentials = Credentials.from_service_account_info(creds_json)
        self.service: Resource = build('drive', 'v3',
                                       credentials=self.credentials)

        # Batch size.
        self.BATCH_SIZE = 4 * mp.cpu_count()

    def download_file(self, file_id: str, output_path: Path):
        """Download a file from Google Drive.

        Args:
            file_id (str): Google Drive item identifier
            output_path (Path): Where to save file
        """
        MAX_TRIES = 3
        COOLDOWN_FACTOR = 3
        next_cooldown = 10
        tries = 0
        success = False
        while not success and tries < MAX_TRIES:
            try:
                request = self.service.files().get_media(fileId=file_id)
                fh = io.FileIO(str(output_path), 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                success = True
            except Exception:
                tries += 1
                if tries < MAX_TRIES:
                    sleep(next_cooldown)
                    next_cooldown *= COOLDOWN_FACTOR
                else:
                    msg = "Failed to download file ID '%s' to path '%s'."
                    msg %= (file_id, str(output_path))
                    print(msg)
                    print(traceback.format_exc())

        return success

    def download_folder(self, folder_id: str, output_dir: Path):
        """Download a folder from Google Drive.

        Args:
            folder_id (str): Google Drive item identifier.
            output_dir (Path): Where to save this folder.
        """
        msg = "Downloading folder '%s' with ID '%s'."
        msg %= (output_dir.name, folder_id)
        print(msg)

        # Create output directory if it doesn't exist.
        output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

        # Fetch files.
        # items: GoogleDriveItem = []
        items = []
        pageToken = ""
        print("Fetching files in folder...")
        while pageToken is not None:
            response = self.service.files().list(
                q=_q_files_in_folder(folder_id),
                pageSize=1000,
                pageToken=pageToken,
                fields="nextPageToken, files(id, name)",
            ).execute()
            items.extend(response.get('files', []))
            pageToken = response.get('nextPageToken')
            print("Files found: %i" % len(items), end="\r")
        _clear_line()

        # Initialize multiprocessing pool.
        with mp.Pool(self.BATCH_SIZE) as p:

            # Download files.
            print("Downloading %i files..." % len(items))
            _print_progress(-1, len(items))
            batch = []
            for idx, file in enumerate(items):
                if output_dir.joinpath(file["name"]).exists():
                    continue

                # Create batch.
                batch.append(file)
                if len(batch) < self.BATCH_SIZE:
                    continue
                _print_progress(idx, len(items))

                # Process batch.
                batch_args = [(item["id"], output_dir.joinpath(item["name"]))
                              for item in batch]
                p.starmap(self.download_file, batch_args)

                # Clear batch.
                batch.clear()

            if len(batch) > 0:
                _print_progress(len(items)-1, len(items))
                # Process batch.
                batch_args = [(item["id"], output_dir.joinpath(item["name"]))
                              for item in batch]
                p.starmap(self.download_file, batch_args)

                # Clear batch.
                batch.clear()
            _clear_line()

        # Fetch folders.
        items = []
        pageToken = ""
        while pageToken is not None:
            response = self.service.files().list(
                q=_q_folders_in_folder(folder_id),
                pageSize=1000,
                pageToken=pageToken,
                fields="nextPageToken, files(id, name)",
            ).execute()
            items.extend(response.get('files', []))
            pageToken = response.get('nextPageToken')

        # Download folders recursively.
        for folder in items:
            sub_dir = output_dir.joinpath(folder["name"])
            self.download_folder(folder["id"], sub_dir)

    # def _process_batch(self, batch: List[GoogleDriveItem], output_dir: Path):

    #     # Construct batch of arguments (file_id, output_path).
    #     batch_args = [(item["id"], output_dir.joinpath(item["name"]))
    #                   for item in batch]

    #     # Run up to CPU_COUNT processes of _download_file() in parallel.
    #     processes = len(batch)
    #     with mp.Pool(processes) as p:
    #         p.starmap(self.download_file, batch_args)
