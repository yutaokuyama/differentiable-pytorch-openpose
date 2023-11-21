from setuptools import setup, find_packages

setup(
    name='differentiable_torch_openpose',
    version="0.0.1",
    description="estimate heatmaps of keypoints with differentiable functions",
    author='okym',
    packages=find_packages(),
    license='MIT' 
)