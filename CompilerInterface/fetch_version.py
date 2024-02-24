import subprocess as sp
import pathlib as pl
import re

version_regex = re.compile(r"^project\(MLCompilerBridge VERSION (?P<version>[^)]+)\)$")
toml_field_regex = r'version[ ]*=[ ]*"(.*)"'

VERSION = ""
with open("../CMakeLists.txt", "r") as f:
    for line in f:
        vmatch = version_regex.match(line)  # Not using walrus because Python3.6
        if vmatch:
            VERSION = vmatch.group("version")
            break

print("Version detected =", VERSION)
lines = []
with open("./pyproject.toml", "r") as f:
    lines = f.readlines()

with open("./pyproject.toml", "w") as f:
    for line in lines:
        if re.search(toml_field_regex, line):
            new_text = f'version = "{VERSION}"\n'
            f.write(new_text)
        else:
            f.write(line)
