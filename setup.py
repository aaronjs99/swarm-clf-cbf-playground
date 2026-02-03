from setuptools import setup, find_packages

setup(
    name="clf_cbf_nav",
    version="0.1.0",
    author="Aaron John Sabu",
    description="A library for CLF-CBF based robotic navigation",
    package_dir={"": "scripts"},
    packages=find_packages(where="scripts"),
    install_requires=["numpy", "matplotlib", "cvxopt"],
    python_requires=">=3.8",
)
