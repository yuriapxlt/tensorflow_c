ARG BASE=ubuntu:20.04
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        wget \
        unzip \
        python3-pip \
        software-properties-common && \
    add-apt-repository -y ppa:openjdk-r/ppa && \
    apt-get update && \
    apt-get install -y \
        openjdk-8-jdk && \
    apt-get clean && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

ARG BAZEL_VERSION=1.1.0
RUN mkdir /bazel && \
    wget --no-check-certificate -O /bazel/installer.sh "https://github.com/bazelbuild/bazel/releases/download/${BAZEL_VERSION}/b\
azel-${BAZEL_VERSION}-installer-linux-x86_64.sh" && \
    chmod +x /bazel/installer.sh && \
    /bazel/installer.sh  && \
    rm -rf /bazel

RUN useradd -rm -d /home/tensorflow -s /bin/bash -g root -G sudo -u 1000 tensorflow
USER tensorflow
WORKDIR /home/tensorflow
