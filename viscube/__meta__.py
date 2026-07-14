"""Package metadata (name, version, author) used by setup.py and __init__."""
# `name` is the name of the package as used for `pip install package`
name = "viscube"
# `path` is the name of the package for `import package`
path = name.lower().replace("-", "_").replace(" ", "_")
# Your version number should follow https://python.org/dev/peps/pep-0440 and
# https://semver.org
version = "0.4.6"
author = "Michael James Yantovski Barth"
author_email = "mjb299@pitt.edu"
description = "Visibility-space gridder for image cubes"  # One-liner
url = ""  # your project homepage
license = "Unlicense"  # See https://choosealicense.com
