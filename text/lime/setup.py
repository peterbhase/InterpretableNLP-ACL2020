from setuptools import setup, find_packages

setup(name='lime',
      version='0.1.1.36',
      description='Local Interpretable Model-Agnostic Explanations for machine learning classifiers',
      url='http://github.com/marcotcr/lime',
      author='Marco Tulio Ribeiro',
      author_email='marcotcr@gmail.com',
      license='BSD',
      packages= find_packages(exclude=['js', 'node_modules', 'tests']),
      install_requires=[
          'matplotlib==2.1.0;python_version<"3.0"',
          'matplotlib;python_version>="3.0"',
          'numpy',
          'scipy',
          'progressbar',
          'scikit-learn>=0.18',
          'scikit-image>=0.12;python_version>="3.0"',
          'scikit-image<0.15;python_version<"3.0"'
      ],
      include_package_data=True,
      zip_safe=False)
