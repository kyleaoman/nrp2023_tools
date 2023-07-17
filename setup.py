from setuptools import setup

setup(
    name="nrp2023_tools",
    version="0.1",
    description="Extra tools for pysphviewer.",
    url="",
    author="Kyle Oman",
    author_email="kyle.a.oman@durham.ac.uk",
    license="",
    packages=["nrp2023_tools"],
    install_requires=[
        "numpy",
        "astropy",
        "pandas",
        "h5py",
        "simfiles",
        "simobj",
        "simtrees",
        "scipy",
    ],
    include_package_data=True,
    zip_safe=False,
)
