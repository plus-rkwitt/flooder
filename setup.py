from setuptools import setup, find_packages

setup(
    name="flooder",
    version="1.0rc6",
    description="Flood complex PH",
    author="Paolo Pellizzoni, Florian Graf, Martin Uray, Stefan Huber, Roland Kwitt",
    author_email="roland.kwitt@gmail.com",
    packages=find_packages(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "flooder = flooder.cli:main",
        ],
    },
)
