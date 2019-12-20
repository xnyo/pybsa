# pybsa
> Experimental BSA packer, written in Python. Currently, it only supports Skyrim Special Edition

## Requirements
- Python 3(.7)
- virtualenv (optional, but recommended)

## Usage
```
$ pip install -r requirements.txt
$ python bsa.py --help
usage: bsa.py [-h] [-z] [-l COMPRESSION_LEVEL] [-d DIRECTORY [DIRECTORY ...]]
              [-f FILE [FILE ...]] [-v]
              root output

Packages loose files into .BSA archives.

positional arguments:
  root                  Path to the data folder root
  output                Output file name

optional arguments:
  -h, --help            show this help message and exit
  -z, --compress        Compresses the output archive
  -l COMPRESSION_LEVEL, --compression-level COMPRESSION_LEVEL
                        Compression level (0-16).0: minimum compression
                        (faster, bigger file), 16: maximum (slower, smaller
                        file). Default: 16.
  -d DIRECTORY [DIRECTORY ...], --directory DIRECTORY [DIRECTORY ...]
                        Subfolders to include in the archive (recursive). They
                        must be paths relative to the archive root (eg:
                        'meshes').Specify more folders separated by a space to
                        pack multiple folders.
  -f FILE [FILE ...], --file FILE [FILE ...]
                        Single files to include in the archive. They must be
                        paths relative to the archive root (eg:
                        'meshes\mymod\mymesh.nif').Specify more files
                        separated by a space to pack multiple files.
  -v, --verbose         Log more information
```

## Examples
Packing chargen, no compression (input size: ~134MB, output size: ~134MB)

```
python bsa.py "C:\ModOrganizer\mods\MyMod" chargen.bsa -d "meshes\actors\character\facegendata\facegeom" -d "textures\actors\character\facegendata\facetint"
INFO:root:Archive flags: Flags.DEFAULT
INFO:root:File flags: FileFlag.MISCELLANEOUS|TEXTURES|MESHES
INFO:root:Folders count: 2
INFO:root:Files count: 181
INFO:root:Compression level: no compression
v Writing BSA file [100%]
INFO:root:Done. Took 0.25499892234802246 seconds.
```

---

Packing chargen, default compression (input size: ~134MB, output size: ~44MB)

```
python bsa.py "C:\ModOrganizer\mods\MyMod" chargen.bsa -d "meshes\actors\character\facegendata\facegeom" -d "textures\actors\character\facegendata\facetint" -z
INFO:root:Archive flags: Flags.DEFAULT|RETAIN_STRINGS_DURING_STARTUP|COMPRESSED|INCLUDE_FILE_NAMES|INCLUDE_DIRECTORY_NAMES
INFO:root:File flags: FileFlag.MISCELLANEOUS|TEXTURES|MESHES
INFO:root:Folders count: 2
INFO:root:Files count: 181
INFO:root:Compression level: 8
v Writing BSA file [100%]
INFO:root:Done. Took 3.5219995975494385 seconds.
```

---

Packing chargen, low compression (input size: ~134MB, output size: ~50MB)

```
python bsa.py "C:\ModOrganizer\mods\MyMod" chargen.bsa -d "meshes\actors\character\facegendata\facegeom" -d "textures\actors\character\facegendata\facetint" -z -l 2
INFO:root:Archive flags: Flags.DEFAULT|RETAIN_STRINGS_DURING_STARTUP|COMPRESSED|INCLUDE_FILE_NAMES|INCLUDE_DIRECTORY_NAMES
INFO:root:File flags: FileFlag.MISCELLANEOUS|TEXTURES|MESHES
INFO:root:Folders count: 2
INFO:root:Files count: 181
INFO:root:Compression level: 2
v Writing BSA file [100%]
INFO:root:Done. Took 0.41900110244750977 seconds.
```

## LICENCE
MIT