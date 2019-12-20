import time
import struct
import argparse
import os
import sys
from enum import Enum, IntFlag, Flag, auto
from typing import List, Iterator, Optional, Set, Any
from typing.io import BinaryIO
import functools
import operator
import logging
import threading

import lz4.frame
from halo import Halo


class PackError(Exception):
    pass


class EmptyError(PackError):
    pass


class UnsupportedFlagsError(PackError):
    pass


class UnsupportedOperationError(PackError):
    pass


class DuplicateFileError(PackError):
    pass


class Game(Enum):
    """Game Enum"""
    LEGENDARY = 0x68
    SPECIAL = 0x69


class Flags(IntFlag):
    """BSA archive flags"""
    INCLUDE_DIRECTORY_NAMES = auto()
    INCLUDE_FILE_NAMES = auto()
    COMPRESSED = auto()
    RETAIN_DIRECTORY_NAMES = auto()
    RETAIL_FILE_NAMES = auto()
    RETAIN_FILE_NAMES_OFFSETS = auto()
    XBOX360 = auto()
    RETAIN_STRINGS_DURING_STARTUP = auto()
    EMBED_FILE_NAMES = auto()
    XMEM = auto()

    DEFAULT = INCLUDE_DIRECTORY_NAMES | INCLUDE_FILE_NAMES | RETAIN_STRINGS_DURING_STARTUP


class FileFlag(IntFlag):
    """BSA file type flags"""
    NONE = 0
    MESHES = auto()
    TEXTURES = auto()
    MENUS = auto()
    SOUNDS = auto()
    VOICES = auto()
    SHADERS = auto()
    TREES = auto()
    FONTS = auto()
    MISCELLANEOUS = auto()

    @staticmethod
    def extension_to_flag(ext: str) -> "FileFlag":
        """Converts an extension (eg: '.dds') to a FileFlag.

        Args:
            ext (str): Extension. Must start with a dot. Case insensitive.
        
        Returns:
            FileFlag: FileFlag that represents the current file
        """
        return {
            ".nif": FileFlag.MESHES,
            ".dds": FileFlag.TEXTURES,
            ".xml": FileFlag.MENUS | FileFlag.MISCELLANEOUS,
            ".wav": FileFlag.SOUNDS,
            ".fuz": FileFlag.SOUNDS,
            ".mp3": FileFlag.VOICES,
            ".ogg": FileFlag.VOICES,
            ".txt": FileFlag.SHADERS,
            ".htm": FileFlag.SHADERS,
            ".bat": FileFlag.SHADERS,
            ".scc": FileFlag.SHADERS,
            ".spt": FileFlag.TREES,
            ".fnt": FileFlag.FONTS,
            ".tex": FileFlag.FONTS,
        }.get(ext.lower(), FileFlag.MISCELLANEOUS)


def tes_hash(file_name: str) -> int:
    """Calculates the BSA hash for the specified file name.
    
    Args:
        file_name (str): name of the file. case-insensitive,
        may contain either slashes or backslashes as path separators (will be sanitized)
    
    Returns:
        int: calculated hash
    """
    root, ext = os.path.splitext(sanitize_path(file_name))
    chars = [ord(x) for x in root]
    # hash1 = chars[-1] | (0, chars[-2])[len(chars) > 2] << 8 | len(chars) << 16 | chars[0] << 24
    hash1 = chars[-1] | (chars[-2] if len(chars) > 2 else 0) << 8 | len(chars) << 16 | chars[0] << 24
    hash1 |= {
        ".kf": 0x80,
        ".nif": 0x8000,
        ".dds": 0x8080,
        ".wav": 0x80000000
    }.get(ext, 0)
    uint_mask, hash2, hash3 = 0xFFFFFFFF, 0, 0 
    for char in chars[1:-2]:
        hash2 = ((hash2 * 0x1003f) + char) & uint_mask
    for char in [ord(x) for x in ext]:
        hash3 = ((hash3 * 0x1003F) + char) & uint_mask
    hash2 = (hash2 + hash3) & uint_mask
    r = (hash2 << 32) + hash1
    logging.debug(f"Hash of {file_name}: {r}")
    return r


def sanitize_path(path: str) -> str:
    """Sanitized a path for hash calculation.
    Basically makes it lowercase and replaces slashes with backslashes.
    
    Args:
        path (str): input path
    
    Returns:
        str: sanitized path
    """
    return path.lower().replace("/", "\\")


def pack_str(s: str) -> bytes:
    """Returns a binary string (length + ascii characters + NULL)
    
    Args:
        s (str): input string
    
    Returns:
        bytes: binary string (length + ascii characters + NULL)
    """
    l = len(s)
    p = struct.pack(f"<b{l}sc", l + 1, s.encode("ascii"), b"\x00")
    return p


class Node:
    """A folder/file inside a BSA archive"""
    def __init__(self, name: str):
        """Initializes a new Node
        
        Args:
            name (str): name of the node
        """
        self.name = name
        self._hash: Optional[int] = None
       
    @property
    def tes_hash(self) -> int:
        """Cached property that returns the tes hash of the current node's name.
        Since the result is cached, it's cheap calling this function multiple times.
        
        Returns:
            int: hash of the current file.
        """
        if self._hash is None:
            self._hash = tes_hash(self.name)
        return self._hash


class Root(Node):
    """Root of the BSA file. Can contain only folders and has no TES hash."""
    def __init__(self):
        super(Root, self).__init__("\\")
        self.subfolders: List[Folder] = []
        self._hash = None


class File(Node):
    """A file inside a BSA archive"""

    def __init__(self, folder: "Folder", name: str):
        """Initialized a file
        
        Args:
            folder (Folder): Folder object this file belongs to
            name (str): name of the file (eg: 'armor.nif')
        """
        super(File, self).__init__(sanitize_path(name))
        self.folder = folder
        self.record_offset: Optional[int] = None

        logging.debug(f"Added file {self.name}")

    @property
    def flags(self) -> FileFlag:
        """BSA file flags based on this file's extension
        
        Returns:
            FileFlag: this file's BSA flags
        """
        _, ext = os.path.splitext(sanitize_path(self.name))
        return FileFlag.extension_to_flag(ext)


class Folder(Node):
    """A folder inside a BSA archive"""

    def __init__(self, bsa: "Bsa", relative_path: str):
        """Initializes a new Folder
        
        Args:
            bsa (Bsa): bsa file this folder belongs to
            relative_path (str): path of this folder, relative to the BSA root (eg: 'meshes\\mymod')
        """
        super(Folder, self).__init__(sanitize_path(relative_path))
        self.bsa = bsa
        self._sorted_files: Optional[List[File]] = None
        logging.debug(f"Added folder {self.name}")

        self.files: List[File] = []
        self.subfolders: List[Folder] = []


    def add_file(self, f: File) -> None:
        """Adds a file to the list of files.
        This invalidates the sorted files cache.
        
        Args:
            f (File): file object to add to this folder
        """
        self.files.append(f)
        # Reset cached sorted files list
        self._sorted_files = None
    
    @property
    def relative_path(self) -> str:
        """Path of this folder, relative to the data path (BSA root). Read only.
        
        Returns:
            str: path of the folder relative to the BSA root
        """
        return self._relative_path

    @property
    def absolute_path(self) -> str:
        """Absolute path of this folder. Read only.
        
        Returns:
            str: absolute path of the folder
        """
        return self._absolute_path

    @property
    def sorted_files(self) -> List["File"]:
        """Cached property that returns all files directly in this folder, sorted by their TES hash.
        
        Returns:
            List[File]: list of files in this folder, sorted by TES hash.
        """
        if self._sorted_files is None:
            self._sorted_files = sorted(self.files, key=lambda x: x.tes_hash)
        return self._sorted_files

    @property
    def empty(self) -> bool:
        """Whether the folder is empty (contains any files, may contain subfolders)
        
        Returns:
            bool: True if the folder contains no files. It may still contain subfolders.
        """
        return not bool(self.files)

    def walk_subfolders(self) -> Iterator["Folder"]:
        """Recursively walks over all subfolders (does not yield the current folder)
        
        Yields:
            Folder: yields all subfolders
        """
        for f in self.subfolders:
            yield f
            yield from f.walk_subfolders()

    def walk_files(self) -> Iterator["File"]:
        """Recursively walks over all files in this folder and its subfolders
        
        Yields:
            File: yields all files in the folder and subfolders
        """
        for f in self.files:
            yield f
        for f in self.subfolders:
            yield from f.walk_files()

    @property
    def files_count(self) -> int:
        """Calculates the number of files in the folder and its subfolders
        
        Returns:
            int: number of files
        """
        return len(self.files) + sum(x.files_count for x in self.subfolders)

    @property
    def flags(self) -> FileFlag:
        """Calculates the file flag based on this folder and its subfolder
        
        Returns:
            FileFlag: the file flag
        """
        return functools.reduce(
            operator.or_, (x.flags for x in self.files), FileFlag.NONE
        ) | functools.reduce(
            operator.or_, (x.flags for x in self.subfolders), FileFlag.NONE
        )
        


class Bsa:
    """A BSA archive"""

    def __init__(
        self, file_name: str, data_path: str, game: Game,
        flags: Flags = Flags.DEFAULT, file_flags: FileFlag = FileFlag.NONE, auto_file_flags: bool = True,
        compress: bool = False, compression_level: int = lz4.frame.COMPRESSIONLEVEL_MAX
    ):
        """Initializes a Bsa object
        
        Args:
            file_name (str): name of the output file
            data_path (str): path of the bsa root folder
            game (Game): game (Skyrim LE/SE). Currently, only SE is supported.
            flags (Flags, optional):    Archive flags.
                                        Flags.DEFAULT will always be forced or the game will not load the archive.
                                        Defaults to Flags.DEFAULT.
            file_flags (FileFlag, optional): File flags. Ignored if auto_file_flags =  True. Defaults to FileFlag.NONE.
            auto_file_flags (bool, optional):   If True, ignore file_flags and set it to the right value based
                                                on the files that will be added to the BSA. Defaults to True.
            compress (bool, optional): Whether the archive should be compressed. Defaults to False.
            compression_level (int, optional):  LZ4 compression level. 0-16, where 0 is min compression, 16 max compression.
                                                Defaults to lz4.frame.COMPRESSIONLEVEL_MAX.
        
        Raises:
            NotImplementedError: if passing game = Game.LEGENDARY
        """
        self.file_name = file_name
        self.game = game
        self.compress = compress
        self.compression_level = compression_level
        self.file_flags: FileFlag = file_flags
        self.flags = flags | Flags.DEFAULT
        if not self.compress:
            self.flags &= ~Flags.COMPRESSED
        else:
            self.flags |= Flags.COMPRESSED
        self.auto_file_flags: bool = auto_file_flags
        self.data_path: str = sanitize_path(data_path)
        self.i = 0

        self._sorted_folders: Optional[List[Folder]] = None
        self.root = Root()
        if game == Game.LEGENDARY:
            raise NotImplementedError("Oldrim is not supported yet")

    @property
    def sorted_folders(self) -> List[Folder]:
        """Returns all folders sorted by their TES hash.
        
        Returns:
            List[Folder]: List of Folder objects, sorted by TES hash.
        """
        if self._sorted_folders is None:
            self._sorted_folders = sorted(
                (x for x in self.walk_folders() if not x.empty),
                key=lambda x: x.tes_hash
            )
        return self._sorted_folders

    def write_header(self, f: BinaryIO) -> None:
        """Writes the BSA header to f
        
        Args:
            f (BinaryIO): file-like output stream
        """
        if self.auto_file_flags:
            logging.debug("Determining file flags")
            self.file_flags = functools.reduce(operator.or_, (x.flags for x in self.walk_folders()), FileFlag.NONE)
        logging.info(f"Archive flags: {str(self.flags)}")
        logging.info(f"File flags: {str(self.file_flags)}")
        logging.info(f"Folders count: {self.non_empty_folders_count}")
        logging.info(f"Files count: {self.files_count}")
        logging.debug(f"Total folder length: {self.total_folder_name_length}")
        logging.debug(f"Total file names length: {self.total_file_name_length}")
        logging.info(f"Compression level: {self.compression_level if self.compress else 'no compression'}")
        logging.debug("Writing header")
        f.write(b"BSA\x00")
        f.write(
            struct.pack(
                "<LLLLLLLL",
                self.game.value,
                0x24,
                self.flags.value,
                self.non_empty_folders_count,
                self.files_count,
                self.total_folder_name_length,
                self.total_file_name_length,
                self.file_flags.value
            )
        )
    
    def write_folder_records(self, f: BinaryIO) -> None:
        """Writes the folder records block to f
        
        Args:
            f (BinaryIO): file-like output stream
        """
        # And write their info
        logging.debug("Writing folder records")
        logging.debug(f"Sorted folder hashes: {[x.tes_hash for x in self.sorted_folders]}")
        for folder in self.sorted_folders:
            folder.record_offset = f.tell()
            f.write(
                struct.pack(
                    "<QLLQ",
                    folder.tes_hash,
                    len(folder.files),
                    0,
                    0
                )
            )

    def write_file_records(self, f: BinaryIO) -> None:
        """Writes the file records block to f
        
        Args:
            f (BinaryIO): file-like output stream
        """
        logging.debug("Writing file records")
        for folder in self.sorted_folders:
            logging.debug(f"Processing file records for folder {folder.name}")
            offset = f.tell()
            f.seek(folder.record_offset + 8 + 4 + 4)
            f.write(struct.pack("<Q", offset + self.total_file_name_length))
            f.seek(0, os.SEEK_END)
            if (self.flags & Flags.INCLUDE_DIRECTORY_NAMES) > 0:
                f.write(pack_str(folder.name))
            logging.debug(f"Sorted files in {folder.name}: {[x.tes_hash for x in folder.sorted_files]}")
            for file in folder.sorted_files:
                file.record_offset = f.tell()
                f.write(
                    struct.pack(
                        "<QLL",
                        file.tes_hash,
                        0,
                        0
                    )
                )
    
    def write_file_names(self, f: BinaryIO) -> None:
        """Writes file names block to f.
        If BSA flags do not have the Flags.INCLUDE_FILE_NAMES set, this method does nothing.
        
        Args:
            f (BinaryIO): file-like output stream
        """
        if (self.flags & Flags.INCLUDE_FILE_NAMES) == 0:
            return
        logging.debug("Writing file names")
        for folder in self.sorted_folders:
            for file in folder.sorted_files:
                f.write(file.name.encode("ascii"))
                f.write(b"\x00")

    def write_files(self, f: BinaryIO) -> None:
        """Writes file data to f.
        
        Args:
            f (BinaryIO): file-like output stream
        """
        # TODO: name 	bstring 	Full path and name of the file. Only present if Bit 9 of archiveFlags is set.
        self.i = 0
        total = self.files_count
        for folder in self.sorted_folders:
            for file in folder.sorted_files:
                p = f"{file.folder.name}\\{file.name}"
                # logging.info(f"Writing {p:100s}[{(i * 100) / total:2.2f}%]")
                data_start = f.tell()
                with open(os.path.join(self.data_path, folder.name, file.name), "rb") as o:
                    if not self.compress:
                        f.write(o.read())
                    else:
                        uncompressed_data = o.read()
                        compressed_data = lz4.frame.compress(uncompressed_data, compression_level=self.compression_level)
                        f.write(struct.pack("<L", len(uncompressed_data)))
                        f.write(compressed_data)
                size = f.tell() - data_start
                f.seek(file.record_offset + 8)
                f.write(struct.pack("<LL", size + (4 if self.compress else 0), data_start))
                f.seek(0, os.SEEK_END)
                self.i += 1

    def write(self):
        """Writes the BSA file. Must be called after adding files/folders to the bsa
        
        Raises:
            EmptyError: if the archive is empty
            UnsupportedFlagsError:  if the archive contains sound files and it's compressed
                                    (can't be loaded by the game for some reason)
        """
        if not self.root.subfolders:
            raise EmptyError("No files present in the BSA.")
        if self.compress and (self.file_flags & (FileFlag.VOICES | FileFlag.SOUNDS)) > 0:
            raise UnsupportedFlagsError(
                "Compressed archives with voices and sounds inside will crash the game."
                "Please disable compression. Aborted."
            )
        with open(self.file_name, "wb") as f:
            self.write_header(f)
            self.write_folder_records(f)
            self.write_file_records(f)
            self.write_file_names(f)
            self.write_files(f)

    def add_folder(self, path: str) -> None:
        """Recursively adds all files in a folder to the archive.
        
        Args:
            path (str): path of the folder, relative to BSA root (eg: 'meshes')
        """
        for root, dirs, files in os.walk(os.path.join(self.data_path, path)):
            root = sanitize_path(root)
            assert root.startswith(self.data_path)
            for file in files:
                self.add_file(os.path.join(root[len(self.data_path) + 1:], file))

    def add_file(self, path: str) -> None:
        """Adds a single file to the archive.
        
        Args:
            path (str): path of the file, relative to BSA root (eg: 'meshes\\mymod\\armor.nif')
        
        Raises:
            DuplicateFileError: if the file is already in the archive
            UnsupportedOperationError: when trying to add a file to the root of the archive
        """
        path = sanitize_path(path)
        path_parts = path.split("\\")
        current = self.root
        path_so_far = []
        for part in path_parts[:-1]:
            path_so_far.append(part)
            path_so_far_str = "\\".join(path_so_far)
            next_ = next((x for x in current.subfolders if x.name == path_so_far_str), None)
            if next_ is None:
                new_folder = Folder(self, path_so_far_str)
                current.subfolders.append(new_folder)
                current = new_folder
            else:
                current = next_
        if current is self.root:
            raise UnsupportedOperationError(f"Can't add files to the root of the archive ({path}), they must be in a folder.")
        if any(x.name == path_parts[-1] for x in current.files):
            raise DuplicateFileError(f"File {path_parts[-1]} already added!")
        current.add_file(File(current, path_parts[-1]))

    @property
    def files_count(self) -> int:
        """Returns the amount of files in the archive
        
        Returns:
            int: number of files in the archive
        """
        return sum(1 for x in self.walk_files())

    @property
    def folders_count(self) -> int:
        """Returns the amount of folders in the archive
        
        Returns:
            int: number of folders in the archive
        """
        return sum(1 for x in self.walk_folders())
    
    @property
    def non_empty_folders_count(self) -> int:
        """Returns the amount of non-empty folders in the archive
        
        Returns:
            int: number of non-empty folders in the archive
        """
        return sum(1 for x in self.walk_folders() if not x.empty)

    @property
    def total_folder_name_length(self) -> int:
        """Returns the length of all folder names inside the archive. Used in BSA header.
        
        Returns:
            int: sum of all folder names + NULL terminators
        """
        # Sum of length of non empty folders
        s = 0
        for x in self.walk_folders():
            if not x.empty:
                s += len(x.name) + 1
        return s

    @property
    def total_file_name_length(self) -> int:
        """Returns the length of all file names inside the archive. Used in BSA header.
        
        Returns:
            int: sum of all file names + NULL terminators
        """
        # +1 is NULL terminator
        return sum(len(x.name) + 1 for x in self.walk_files())

    def walk_folders(self) -> Iterator[Folder]:
        """Recursively walks over all folders in the archive.
        
        Yields:
            Iterator[Folder]: Folder object
        """
        for x in self.root.subfolders:
            yield x
            yield from x.walk_subfolders()

    def walk_files(self) -> Iterator[File]:
        """Recursively walks over all files in the archive.
        
        Yields:
            Iterator[File]: File objects
        """
        for folder in self.root.subfolders:
            yield from folder.walk_files()


def main():
    parser = argparse.ArgumentParser(description="Packages loose files into .BSA archives.")
    parser.add_argument("root", help="Path to the data folder root")
    parser.add_argument("output", help="Output file name")
    parser.add_argument(
        "-z",
        "--compress",
        action="store_true",
        help="Compresses the output archive",
        default=False,
        required=False
    )
    parser.add_argument(
        "-l",
        "--compression-level",
        type=int,
        help=\
            "Compression level (0-16)."
            "0: minimum compression (faster, bigger file), "
            "16: maximum (slower, smaller file). "
            "Default: 16.",
        default=8,
        required=False
    )
    parser.add_argument(
        "-d",
        "--directory",
        nargs="+",
        help="Subfolders to include in the archive (recursive). "
             "They must be paths relative to the archive root (eg: 'meshes')."
             "Specify more folders separated by a "
             "space to pack multiple folders.",
        required=False,
        default=[]
    )
    parser.add_argument(
        "-f",
        "--file",
        nargs="+",
        help="Single files to include in the archive. "
             "They must be paths relative to the archive root (eg: 'meshes\\mymod\\mymesh.nif')."
             "Specify more files separated by a "
             "space to pack multiple files.",
        required=False,
        default=[]
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Log more information",
        default=False,
        required=False
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.compression_level < 0 or args.compression_level > 16:
        raise ValueError("Compression must be between 0 and 16.")

    st = time.time()
    archive = Bsa(args.output, data_path=args.root, game=Game.SPECIAL, compress=args.compress, compression_level=args.compression_level)
    folders = []
    if not args.directory and not args.file:
        for x in os.listdir(args.root):
            if os.path.isdir(os.path.join(os.path.abspath(args.root), x)):
                folders.append(x)
    else:
        folders = args.directory

    try:
        for folder in folders:
            archive.add_folder(folder)

        for file_ in args.file:
            archive.add_file(file_)
        spinner = Halo(text="Writing BSA file", spinner={
            "interval": 80,
            "frames": [
                "⠋",
                "⠙",
                "⠹",
                "⠸",
                "⠼",
                "⠴",
                "⠦",
                "⠧",
                "⠇",
                "⠏"
            ]
        })
        spinner.start()
        def update_spinner():
            perc = 0
            total = archive.files_count
            while True:
                perc = (archive.i * 100) / total
                spinner.text = f"Writing BSA file [{perc:2.2f}%]"
                time.sleep(0.5)
        t = threading.Thread(target=update_spinner, daemon=True)
        t.start()
        archive.write()
        spinner.text = f"Writing BSA file [100%]"
        spinner.succeed()
    except PackError as e:
        logging.error(e)
    et = time.time()
    logging.info(f"Done. Took {et - st} seconds.")
    

if __name__ == "__main__":
    main()
