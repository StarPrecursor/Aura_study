import setuptools

setuptools.setup(
    name="artool",
    version="0.0.1",
    auther="Zhe Yang",
    auther_email="starprecursor@gmail.com",
    description="Tools for Aura coin research",
    #url=None,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9, <3.10",
    install_requires=[
        "pyarrow",
        "scikit-learn",
        "pandas",
        "scipy",
        "matplotlib",
        "PyYAML",
    ],
    entry_points={
        #"console_scripts": ["",],
    },
)
