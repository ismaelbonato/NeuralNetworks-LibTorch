FROM ubuntu:26.04

ARG LIBTORCH_URL=https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip

# Install all development tools and dependencies in a single layer
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    unzip \
    software-properties-common \
    gnupg \
    ca-certificates \
    git \
    cmake \
    ninja-build \
    cppcheck \
    catch2 \
    sudo \
    gcc-14 \
    g++-14 \
    clang-20 \
    clangd-20 \
    clang-format-20 \
    clang-tidy-20 \
    libc++-20-dev \
    libc++abi-20-dev \
    gdb \
    lldb-20 \
    valgrind \
    vim \
    nano \
    lcov \
    gcovr \
    pkg-config \
    libopencv-dev \
    libomp-dev \
    libgomp1 \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 \
    && update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100 \
    && update-alternatives --install /usr/bin/clangd clangd /usr/bin/clangd-20 100 \
    && update-alternatives --install /usr/bin/clang-format clang-format /usr/bin/clang-format-20 100 \
    && update-alternatives --install /usr/bin/clang-tidy clang-tidy /usr/bin/clang-tidy-20 100 \
    && update-alternatives --install /usr/bin/lldb lldb /usr/bin/lldb-20 100 \
    && update-alternatives --install /usr/bin/gcov gcov /usr/bin/gcov-14 100 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget -qO /tmp/libtorch.zip "${LIBTORCH_URL}" \
    && unzip -q /tmp/libtorch.zip -d /usr/local \
    && rm /tmp/libtorch.zip

# Verify installations
RUN gcc --version && g++ --version && clang --version && clang++ --version && cmake --version && test -f /usr/local/libtorch/share/cmake/Torch/TorchConfig.cmake

# Declare build args for dynamic UID/GID
ARG USERNAME
ARG UID
ARG GID

# Remove default 'ubuntu' user to avoid UID/GID conflicts when creating a new user matching the host
RUN userdel -r ubuntu || true

# Create a user matching host UID/GID
RUN groupadd --gid ${GID} ${USERNAME} || true && \
    useradd \
      --create-home \
      --gid ${GID} \
      --groups sudo \
      --no-log-init \
      --shell /bin/bash \
      --uid ${UID} \
      ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/${USERNAME}/workspace && \
    chown -R ${UID}:${GID} /home/${USERNAME} && \
    sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' /home/${USERNAME}/.bashrc

USER ${USERNAME}
WORKDIR /home/${USERNAME}/workspace

# Default command
CMD ["/bin/bash"]
