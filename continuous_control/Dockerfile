FROM nvidia/cuda:9.2-cudnn7-runtime-ubuntu18.04

SHELL ["/bin/bash", "--login", "-c"]

# Adapted from: https://github.com/conda/conda-docker/blob/master/miniconda3/debian/Dockerfile
RUN apt-get -qq update && apt-get -qq -y install wget bzip2 git unzip \
    && wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes \
    && conda init bash

ENV PATH /opt/conda/bin:$PATH

WORKDIR /usr/src/app

RUN conda create --name drlnd python=3.6 notebook jedi=0.17.2 -y
SHELL ["conda", "run", "--no-capture-output", "-n", "drlnd", "/bin/bash", "-c"]

RUN git clone https://github.com/udacity/deep-reinforcement-learning.git \
    && cd deep-reinforcement-learning/python \
    && pip install --user . \
    && python -m ipykernel install --user --name drlnd --display-name "drlnd"

RUN wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip \
    && unzip Reacher_Linux_NoVis.zip \
    && rm Reacher_Linux_NoVis.zip \
    && apt-get -y remove wget unzip git

COPY Continuous_Control.ipynb ./

CMD ["conda", "run", "--no-capture-output", "-n", "drlnd", "jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
