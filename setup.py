# from setuptools import setup

# setup(
#     name='Viz_EMNLP',
#     version='',
#     packages=['diffusion', 'diffusion.experiments'],
#     url='',
#     license='',
#     author='amankesarwani',
#     author_email='',
#     description=''
# )

from pathlib import Path
from setuptools import setup, find_packages

this_dir = Path(__file__).parent

setup(
    name="viz_emnlp",
    version="0.0.1",
    description="Diffusion-model experiments for EMNLP visualizations",
    long_description=(this_dir / "README.md").read_text(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    packages=find_packages(include=["diffusion*", "project*"]),
    install_requires=[
        "torch",
        "diffusers",
        "accelerate",
        "pandas",
        "numpy",
        "scikit-learn",
        # add any others you import
    ],
    include_package_data=True,
)

