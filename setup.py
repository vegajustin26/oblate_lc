from setuptools import setup, find_packages

setup(name = "oblate_lc",
      package_dir = {"": "src"},
      packages = find_packages('src', include = ["oblate_lc"]),
      setup_requires=["numpy"],  # Just numpy here
    install_requires=["numpy", "matplotlib"],  # Add any of your other dependencies here
    )