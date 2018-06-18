from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os

# to publish use:
# > python setup.py sdist upload
# which depends on ~/.pypirc

# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            setattr(__builtins__, "__NUMPY_SETUP__", False)
        import numpy
        print("numpy.get_include()", numpy.get_include())
        self.include_dirs.append(numpy.get_include())

def run_setup(with_binary=True, test_xgboost=True, test_lightgbm=True):
    print("run_setup(with_binary=%r, test_xgboost=%r, test_lightgbm=%r)" % (with_binary, test_xgboost, test_lightgbm))

    ext_modules = []
    if with_binary:
        ext_modules.append(
            Extension('shap._cext', sources=['shap/_cext.cc'])
        )

    with open("README.md", "r") as fh:
        long_description = fh.read()

    if test_xgboost and test_lightgbm:
        tests_require = ['nose', 'xgboost', 'lightgbm']
    elif test_xgboost:
        tests_require = ['nose', 'xgboost']
    elif test_lightgbm:
        tests_require = ['nose', 'lightgbm']
    else:
        tests_require = ['nose']

    print("tests_require = %r" % tests_require)

    setup(
        name='shap',
        version='0.19.0',
        description='A unified approach to explain the output of any machine learning model.',
        long_description=long_description,
        long_description_content_type="text/markdown",
        url='http://github.com/slundberg/shap',
        author='Scott Lundberg',
        author_email='slund1@cs.washington.edu',
        license='MIT',
        packages=['shap', 'shap.explainers'],
        cmdclass={'build_ext': build_ext},
        setup_requires=['numpy'],
        install_requires=['numpy', 'scipy', 'iml>=0.6.0', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm'],
        test_suite='nose.collector',
        tests_require=tests_require,
        ext_modules=ext_modules,
        zip_safe=False
    )
    print("Setup is complete!")

# xgboost can't be installed from pip on windows
if __name__ == "__main__":
    if os.name == 'nt':
        print("Building on Windows, so skipping XGBoost tests...")
        try:
            run_setup(with_binary=True, test_xgboost=False, test_lightgbm=True)
        except Exception as e:
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            run_setup(with_binary=False, test_xgboost=False, test_lightgbm=True)
    else:
        try:
            run_setup(with_binary=True, test_xgboost=True, test_lightgbm=True)
        except Exception as e:
            print("WARNING: The C extension could not be compiled, sklearn tree models not supported.")
            run_setup(with_binary=False, test_xgboost=True, test_lightgbm=True)
