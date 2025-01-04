#####################################
# RCP CaaS requirement (Image)
#####################################
# The best practice is to use an image
# with GPU support pre-built by Nvidia.
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/

# For example, if you want to use an image with pytorch already installed
# FROM --platform=linux/amd64 nvcr.io/nvidia/pytorch:23.11-py3

# IMPORTANT
# The --platform parameter is mandatory on ARM MacOS
# to force the build of the container using amd64 (x64).
# Without this parameter, the container will not work on the CaaS cluster.

# TUTORIAL ONLY
# In this example we'll use a smaller image to speed up the build process.
FROM --platform=linux/arm64/v8 docker.io/library/ubuntu:22.04

#####################################
# RCP CaaS requirement (Storage)
#####################################
# Create your user inside the container.
# This block is needed to correctly map
# your EPFL user id inside the container.
# Without this mapping, you are not able
# to access files from the external storage.
ARG LDAP_USERNAME
ARG LDAP_UID
ARG LDAP_GROUPNAME
ARG LDAP_GID
RUN groupadd ${LDAP_GROUPNAME} --gid ${LDAP_GID}
RUN useradd -m -s /bin/bash -g ${LDAP_GROUPNAME} -u ${LDAP_UID} ${LDAP_USERNAME}
#####################################

# Copy your code inside the container
RUN mkdir -p /home/${LDAP_USERNAME}
COPY ./ /home/${LDAP_USERNAME}

# Set your user as owner of the new copied files
RUN chown -R ${LDAP_USERNAME}:${LDAP_GROUPNAME} /home/${LDAP_USERNAME}

# Install required packages
RUN apt update
RUN apt install python3-pip -y

# Set the working directory in your user's home
WORKDIR /home/${LDAP_USERNAME}
USER ${LDAP_USERNAME}

# Install additional dependencies
RUN pip install -r requirements.txt

