# This Dockerfile builds OpenMS, the TOPP tools, pyOpenMS and thidparty tools.
# It also adds a basic streamlit server that serves a pyOpenMS-based app.
# hints:
# build image and give it a name (here: streamlitapp) with: docker build --no-cache -t streamlitapp:latest --build-arg GITHUB_TOKEN=<your-github-token> . 2>&1 | tee build.log
# check if image was build: docker image ls
# run container: docker run -p 8501:8501 streamlitappsimple:latest
# debug container after build (comment out ENTRYPOINT) and run container with interactive /bin/bash shell
# prune unused images/etc. to free disc space (e.g. might be needed on gitpod). Use with care.: docker system prune --all --force

FROM ubuntu:24.04 AS setup-build-system
# ARG OPENMS_REPO=https://github.com/OpenMS/OpenMS.git
# ARG OPENMS_BRANCH=release/3.5.0
ARG OPENMS_REPO=https://github.com/singjc/OpenMS.git
ARG OPENMS_BRANCH=for/opendiakiosk
ARG PORT=8501
# Streamlit app GitHub user name (to download artifact from).
ARG GITHUB_USER=OpenMS
# Streamlit app GitHub repository name (to download artifact from).
ARG GITHUB_REPO=OpenDIAKiosk

USER root

# Install required Ubuntu packages.
RUN apt-get -y update
RUN apt-get install -y --no-install-recommends --no-install-suggests g++ autoconf automake patch libtool make git gpg wget ca-certificates curl jq libgtk2.0-dev openjdk-8-jdk cron
RUN update-ca-certificates
RUN apt-get install -y --no-install-recommends --no-install-suggests libsvm-dev libeigen3-dev coinor-libcbc-dev libglpk-dev libzip-dev zlib1g-dev libxerces-c-dev libbz2-dev libomp-dev libhdf5-dev patchelf
RUN apt-get install -y --no-install-recommends --no-install-suggests libboost-date-time1.83-dev \
                                                                     libboost-iostreams1.83-dev \
                                                                     libboost-regex1.83-dev \
                                                                     libboost-math1.83-dev \
                                                                     libboost-random1.83-dev
RUN apt-get install -y --no-install-recommends --no-install-suggests qt6-base-dev libqt6svg6-dev libqt6opengl6-dev libqt6openglwidgets6 libgl-dev

RUN set -eux; \
        apt-get update; \
        wget -qO /tmp/apache-arrow-apt-source-latest-noble.deb \
            https://repo1.maven.org/maven2/org/apache/arrow/ubuntu/apache-arrow-apt-source-latest-noble.deb; \
        apt-get install -y --no-install-recommends /tmp/apache-arrow-apt-source-latest-noble.deb; \
        apt-get update; \
        # Pin Arrow 23: find a libparquet-dev candidate that starts with 23.
        ARROW_VER=$(apt-cache madison libparquet-dev | awk '{print $3}' | grep -E '^23\.' | head -n1) || true; \
        if [ -z "$ARROW_VER" ]; then \
                echo "ERROR: no libparquet-dev 23.* available from apt source"; \
                apt-cache madison libparquet-dev || true; \
                exit 1; \
        fi; \
        apt-get install -y --no-install-recommends libparquet-dev="$ARROW_VER" libarrow-dev="$ARROW_VER"; \
        rm -f /tmp/apache-arrow-apt-source-latest-noble.deb; \
        rm -rf /var/lib/apt/lists/*

# Install Github CLI
RUN (type -p wget >/dev/null || (apt-get update && apt-get install wget -y)) \
	&& mkdir -p -m 755 /etc/apt/keyrings \
	&& wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt-get update \
	&& apt-get install gh -y

# Download and install miniforge.
ENV PATH="/root/miniforge3/bin:${PATH}"
RUN wget -q \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b \
    && rm -f Miniforge3-Linux-x86_64.sh
RUN mamba --version

# Make /root traversable so the entrypoint can `source
# /root/miniforge3/bin/activate ...` when the container runs as a non-root
# user (apptainer/singularity maps the host UID into the container; the
# default ubuntu /root is 0700 which would block path traversal). +x only,
# not +r, so the directory listing remains private.
RUN chmod o+x /root

# Setup mamba environment.
RUN mamba create -n streamlit-env python=3.11
RUN echo "mamba activate streamlit-env" >> ~/.bashrc
ENV STREAMLIT_ENV_PREFIX=/root/miniforge3/envs/streamlit-env
SHELL ["/bin/bash", "--rcfile", "~/.bashrc"]
SHELL ["mamba", "run", "-n", "streamlit-env", "/bin/bash", "-c"]

# Install up-to-date cmake via mamba and packages for pyOpenMS build.
RUN mamba install cmake
RUN pip install --upgrade pip && python -m pip install -U setuptools nose cython "autowrap<=0.24" pandas numpy pytest

# Clone OpenMS branch and the associcated contrib+thirdparties+pyOpenMS-doc submodules.
RUN git clone --recursive --depth=1 -b ${OPENMS_BRANCH} --single-branch ${OPENMS_REPO} && cd /OpenMS

# Pull Linux compatible third-party dependencies and store them in directory thirdparty.
WORKDIR /OpenMS
RUN mkdir /thirdparty && \
    git submodule update --init THIRDPARTY && \
    cp -r THIRDPARTY/All/* /thirdparty && \
    cp -r THIRDPARTY/Linux/x86_64/* /thirdparty && \
    chmod -R +x /thirdparty
ENV PATH="/thirdparty/LuciPHOr2:/thirdparty/MSGFPlus:/thirdparty/Sirius:/thirdparty/ThermoRawFileParser:/thirdparty/Comet:/thirdparty/Fido:/thirdparty/MaRaCluster:/thirdparty/MyriMatch:/thirdparty/OMSSA:/thirdparty/Percolator:/thirdparty/SpectraST:/thirdparty/XTandem:/thirdparty/crux:${PATH}"

# Build OpenMS and pyOpenMS.
FROM setup-build-system AS compile-openms
WORKDIR /

# Set up build directory.
RUN mkdir /openms-build
WORKDIR /openms-build

# Configure.
RUN /bin/bash -c "cmake -DCMAKE_BUILD_TYPE='Release' -DCMAKE_PREFIX_PATH='/OpenMS/contrib-build/;/usr/;/usr/local' -DHAS_XSERVER=OFF -DBOOST_USE_STATIC=OFF -DPYOPENMS=ON -DWITH_UV=OFF -DPYOPENMS_PREPARE_WHEEL_REPAIR=ON -DPython_EXECUTABLE=/root/miniforge3/envs/streamlit-env/bin/python ../OpenMS -DPY_MEMLEAK_DISABLE=On"

# Build TOPP tools and clean up.
RUN make -j4 OpenSwathAssayGenerator OpenSwathDecoyGenerator OpenSwathWorkflow
RUN rm -rf src doc CMakeFiles

# Build pyOpenMS and produce a repairable wheel using the CMake packaging target.
RUN make -j4 pyopenms
WORKDIR /openms-build/pyOpenMS

# Ensure wheel tooling is available in the build Python, then package and repair the wheel.
RUN set -eux; \
    PY=/root/miniforge3/envs/streamlit-env/bin/python; \
    echo "Installing wheel tooling into build python..."; \
    $PY -m pip install --no-cache-dir -U pip build auditwheel py-build-cmake; \
    cd /openms-build; \
    echo "Invoking CMake pyopenms_wheel target to package wheel (will continue on error)..."; \
    cmake --build . --target pyopenms_wheel || true; \
    echo "Wheel directory listing (pyopenms_wheels):"; ls -la pyopenms_wheels || true; \
    if compgen -G "pyopenms_wheels/*.whl" > /dev/null; then \
        echo "Found built wheel(s) in pyopenms_wheels, repairing with auditwheel..."; \
        mkdir -p /openms-build/pyopenms_wheels_repaired; \
        auditwheel repair -w /openms-build/pyopenms_wheels_repaired pyopenms_wheels/*.whl; \
        echo "Repaired wheels:"; ls -la /openms-build/pyopenms_wheels_repaired; \
        if compgen -G "/openms-build/pyopenms_wheels_repaired/*.whl" > /dev/null; then \
            echo "Installing repaired wheel(s) into build python..."; \
            for f in /openms-build/pyopenms_wheels_repaired/*.whl; do $PY -m pip install "$f"; done; \
        else \
            echo "ERROR: auditwheel did not produce any repaired wheels"; \
            ls -la /openms-build/pyopenms_wheels_repaired || true; \
            exit 1; \
        fi; \
    elif compgen -G "pyOpenMS/dist/*.whl" > /dev/null; then \
        echo "Found legacy dist wheel, installing..."; \
        $PY -m pip install pyOpenMS/dist/*.whl; \
    else \
        echo "No wheel produced; falling back to development install (editable)"; \
        cd /openms-build/pyOpenMS; \
        echo "Installing editable pyopenms into build python..."; \
        $PY -m pip install -e . --no-cache-dir --no-binary=pyopenms; \
        echo "Editable install completed."; \
    fi

# Check to see if pyOpenMS can be imported and print its version.
RUN set -eux; \
    export LD_LIBRARY_PATH="${STREAMLIT_ENV_PREFIX}/lib:/openms-build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; \
    source /root/miniforge3/bin/activate streamlit-env; \
    python -c "import pyopenms; print('pyOpenMS version:', pyopenms.__version__)"

# Install other dependencies (excluding pyopenms)
COPY requirements.txt ./requirements.txt 
RUN grep -Ev '^pyopenms([=<>!~].*)?$' requirements.txt > requirements_cleaned.txt && mv requirements_cleaned.txt requirements.txt
RUN pip install -r requirements.txt

# Check to see if the same pyOpenMS version is still installed after installing other dependencies.
RUN set -eux; \
    export LD_LIBRARY_PATH="${STREAMLIT_ENV_PREFIX}/lib:/openms-build/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; \
    source /root/miniforge3/bin/activate streamlit-env; \
    python -c "import pyopenms; print('pyOpenMS version after installing other dependencies:', pyopenms.__version__)"


WORKDIR /
RUN mkdir openms

# Copy TOPP tools bin directory, add to PATH.
RUN cp -r openms-build/bin /openms/bin
ENV PATH="/openms/bin/:${PATH}"

# Copy TOPP tools bin directory, add to PATH.
RUN cp -r openms-build/lib /openms/lib
ENV LD_LIBRARY_PATH="${STREAMLIT_ENV_PREFIX}/lib:/openms/lib"

# Copy share folder, add to PATH, remove source directory.
RUN cp -r OpenMS/share/OpenMS /openms/share
RUN rm -rf OpenMS
ENV OPENMS_DATA_PATH="/openms/share/"

# Remove build directory.
RUN rm -rf openms-build

# Prepare and run streamlit app.
FROM compile-openms AS run-app

# Install Redis server for job queue and nginx for load balancing.
# Redis data lives under $RUNTIME_DIR at runtime (see entrypoint.sh) so no
# /var/lib/redis setup is needed - that path is not writable under Apptainer.
RUN apt-get update && apt-get install -y --no-install-recommends redis-server nginx \
    && rm -rf /var/lib/apt/lists/*

# Create Redis data directory. Default 0755 root-owned is enough: the docker
# entrypoint runs as root (can write regardless of mode), and the apptainer
# entrypoint relocates Redis state to /tmp/openms-runtime-* so this dir is
# never written under apptainer.
RUN mkdir -p /var/lib/redis

# Pre-create bind-mount targets so apptainer/singularity has a real attach
# point. Docker auto-creates missing `-v` targets, but singularity uses a
# read-only underlay and silently ignores `:rw` when the target isn't a
# real directory in the SIF — writes then fail with EROFS even though the
# host bind path is writable. Pre-creating these directories costs one
# inode each and changes nothing in docker mode (the user's volume mount
# shadows them).
RUN mkdir -p /workspaces-streamlit-template /mounted-data

# Create workdir and copy over all streamlit related files/folders.

# note: specifying folder with slash as suffix and repeating the folder name seems important to preserve directory structure
WORKDIR /app
COPY assets/ /app/assets
COPY content/ /app/content
COPY docs/ /app/docs
COPY example-data/ /app/example-data
COPY gdpr_consent/ /app/gdpr_consent
COPY hooks/ /app/hooks
COPY src/ /app/src
COPY utils/ /app/utils
COPY app.py /app/app.py
COPY settings.json /app/settings.json
COPY default-parameters.json /app/default-parameters.json
COPY presets.json /app/presets.json
COPY data /app/data

# Set environment variable for Redeem pretrain models
ENV REDEEM_PRETRAINED_MODELS_DIR=/app/data/pretrained_models

# For streamlit configuration
COPY .streamlit/ /app/.streamlit/
COPY clean-up-workspaces.py /app/clean-up-workspaces.py

# add cron job to the crontab
RUN echo "0 3 * * * /root/miniforge3/envs/streamlit-env/bin/python /app/clean-up-workspaces.py >> /app/clean-up-workspaces.log 2>&1" | crontab -

# Set default worker count (can be overridden via environment variable)
ENV RQ_WORKER_COUNT=1
ENV REDIS_URL=redis://localhost:6379/0

# Number of Streamlit server instances for load balancing (default: 1 = no load balancer)
# Set to >1 to enable nginx load balancer with multiple Streamlit instances
ENV STREAMLIT_SERVER_COUNT=1

# Install the apptainer-compatible entrypoint that starts cron (when the root
# FS is writable), Redis, RQ workers, optional nginx load balancer, and the
# Streamlit server. The script falls back to /tmp paths under apptainer.
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Patch Analytics
RUN mamba run -n streamlit-env python hooks/hook-analytics.py

# Set Online Deployment
RUN jq '.online_deployment = true' settings.json > tmp.json && mv tmp.json settings.json

# Point the in-app mounted-drive browser at the conventional bind-mount path.
# The browser only renders when this directory exists at runtime, i.e. when
# the user starts the container with `-v /host/path:/mounted-data`.
RUN jq '.local_data_dir = "/mounted-data"' settings.json > tmp.json && mv tmp.json settings.json

# Download latest OpenMS App executable as a ZIP file.
# ARG declared here (not at the top) — otherwise the per-run token busts the cache.
ARG GITHUB_TOKEN
RUN if [ -n "$GITHUB_TOKEN" ]; then \
        echo "GITHUB_TOKEN is set, proceeding to download the release asset..."; \
        gh release download -R ${GITHUB_USER}/${GITHUB_REPO} -p "OpenDIAKiosk.zip" -D /app; \
    else \
        echo "GITHUB_TOKEN is not set, skipping the release asset download."; \
    fi


# Run app as container entrypoint.
EXPOSE $PORT
ENTRYPOINT ["/app/entrypoint.sh"]
