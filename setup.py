from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="mindcare-ai",
    version="1.0.0",
    author="Rohit Khadka",
    author_email="rohitkhadka153@gmail.com",
    description="AI-powered mental health support chatbot with voice capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohitkhadka1/End-to-End-Medical-Chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    keywords="mental health, chatbot, ai, nlp, healthcare, voice assistant",
    project_urls={
        "Bug Reports": "https://github.com/rohitkhadka1/End-to-End-Medical-Chatbot/issues",
        "Source": "https://github.com/rohitkhadka1/End-to-End-Medical-Chatbot",
    },
)