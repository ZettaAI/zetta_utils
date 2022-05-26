import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ztutil",
    version="0.0.0",
    author="Zetta AI",
    author_email="",
    description="ZettaAI Utility Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    package_data={'': ['*.py']},
    install_requires=[
        'numpy',
        'matplotlib',
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.7',
)
