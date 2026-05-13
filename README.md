<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/OpenMS/OpenDIAKiosk/raw/main/assets/OpenDIAKiosk_logo.png" alt="OpenDIAKiosk_Logo" width="500">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/OpenMS/OpenDIAKiosk/raw/main/assets/OpenDIAKiosk_logo.png" alt="OpenDIAKiosk_Logo" width="500">
    <img alt="OpenDIAKiosk_Logo" comment="Placeholder to transition between light color mode and dark color mode - this image is not directly used." src="https://github.com/OpenMS/OpenDIAKiosk/raw/main/assets/OpenDIAKiosk_logo.png">
  </picture>
</p>

---

# OpenDIAKiosk: A Streamlit app for Information and Tools on All Things DIA

[![Open Template!](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://abi-services.cs.uni-tuebingen.de/streamlit-template/)

This repository is based on the [streamlit-template](https://github.com/OpenMS/streamlit-template) and provides a Streamlit web app for interactive teaching of Data-Independent Acquisition (DIA) concepts as well as practical tools used in DIA analysis. 

## Features

- Workspaces for user data with unique shareable IDs
- Persistent parameters and input files within a workspace
- local and online mode
- Captcha control
- Packaged executables for Windows
- framework for workflows with OpenMS TOPP tools
- Deployment [with docker-compose](https://github.com/OpenMS/streamlit-deployment)

## 🔗 Try the Online Demo

Explore the hosted version here:  👉 [Live App](https://abi-services.cs.uni-tuebingen.de/streamlit-template/)

## 🚀 Self-hosted deployment

Run OpenDIAKiosk on your own server, VM, or workstation by pulling the prebuilt Docker image from GitHub Container Registry — no clone or build step required.

> **Note:** This section covers single-host Docker deployments. For a real Kubernetes cluster (Traefik ingress, Redis-backed job queue, shared PVC), see [`docs/kubernetes-deployment.md`](docs/kubernetes-deployment.md).

### 1. Pull the image

```bash
docker pull ghcr.io/openms/opendiakiosk:latest
```

### 2. Run the container

```bash
docker run -d \
  --name opendiakiosk \
  -p 8501:8501 \
  -v /path/to/data:/mounted-data:ro \
  -v /path/to/workspaces:/workspaces-streamlit-template \
  ghcr.io/openms/opendiakiosk:latest
```

- **`-p 8501:8501`** — exposes the in-container Streamlit server (port 8501) on the host's port 8501. Change the left-hand number to bind to a different host port (e.g. `-p 9000:8501` to reach the UI at `http://host:9000`).

- **`-v /path/to/data:/mounted-data:ro`** — *optional* bind-mount that makes a host directory of MS data files (mzML, raw, etc.) available inside the container at `/mounted-data`. When this directory exists at container start, the in-app upload page auto-detects it and shows an in-app tree browser; selected files are referenced in place (no copy into the workspace volume), so the mount can safely be read-only (`:ro`). Omit this flag entirely to fall back to the standard browser-upload UI.

- **`-v /path/to/workspaces:/workspaces-streamlit-template`** — bind-mount on the host that persists every user workspace (parameters, uploaded inputs, workflow results) across container restarts and upgrades. Without it, all workspaces are lost as soon as the container is removed. Point the left-hand path at any directory on the host with enough free space for the expected workload.

### 3. Access remotely via SSH tunnel

If the host has no public IP — or port 8501 is firewalled (recommended) — forward the port over SSH from your laptop:

```bash
ssh -L 8501:localhost:8501 user@your-server
```

Then open <http://localhost:8501> in your local browser. Add `-N` to skip opening a shell when you only need the tunnel:

```bash
ssh -N -L 8501:localhost:8501 user@your-server
```

### 4. Update to a new version

```bash
docker pull ghcr.io/openms/opendiakiosk:latest
docker stop opendiakiosk && docker rm opendiakiosk
# then re-run the `docker run` command from step 2
```

The host directory bound to `/workspaces-streamlit-template` is untouched by `docker rm`, so all user workspaces are preserved across upgrades.

## 🛰️ Run with Apptainer / Singularity (HPC clusters)

On HPC clusters where Docker isn't available, run the same OCI image under
[Apptainer](https://apptainer.org/) (formerly Singularity) — typically the
default container runtime there. Apptainer runs containers with a **read-only
root filesystem** by default and maps your host UID into the container; the
OpenDIAKiosk entrypoint detects both conditions and falls back to writable
paths under `/tmp` for Redis state and PID files, so no `--writable-tmpfs`
flag is required.

### 1. Pull and convert the image to SIF

```bash
apptainer pull docker://ghcr.io/openms/opendiakiosk:latest
# produces opendiakiosk_latest.sif in the current directory
```

### 2. Run the container

```bash
apptainer run \
  --bind /path/to/data:/mounted-data:ro \
  --bind /path/to/workspaces:/workspaces-streamlit-template \
  opendiakiosk_latest.sif
```

The bind semantics match the Docker `-v` flags above:

- `--bind /path/to/data:/mounted-data:ro` — *optional* read-only mount of the
  MS data directory the in-app file browser will list.
- `--bind /path/to/workspaces:/workspaces-streamlit-template` — persistent
  workspace storage, equivalent to the Docker `-v` for the same target path.

### 3. Notes on apptainer-mode behavior

- The entrypoint logs `Detected read-only root filesystem
  (apptainer/singularity mode)` and routes Redis state to
  `/tmp/openms-runtime-$$` (always writable in apptainer's default tmpfs).
- The workspace-cleanup cron job is **skipped** under apptainer (cron cannot
  write its PID file on a read-only root). If you need periodic cleanup, run
  `clean-up-workspaces.py` from a host-side scheduler instead.
- Network namespaces are shared with the host by default, so port `8501`
  inside the container is reachable on the host immediately — no
  port-mapping flag is required.

## 💻 Run Locally

To run the app locally:

1. **Clone the repository**
   ```bash
   git clone git@github.com:OpenMS/OpenDIAKiosk.git
   cd streamlit-template
   ```

2. **Install dependencies**
   
   Make sure you can run ```pip``` commands.
   
   Install all dependencies with:
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the app**
   ```bash
   streamlit run app.py
   ```

> ⚠️ Note: The local version offers limited functionality. Features that depend on OpenMS TOPP tools are only available out of the box in the Docker setup. For the local version [OpenMS Command Line Tools](https://openms.readthedocs.io/en/latest/about/installation.html) must be installed separately.


## 🐳 Build with Docker

This repository contains two Dockerfiles.

1. `Dockerfile`: This Dockerfile builds all dependencies for the app including Python packages and the OpenMS TOPP tools. Recommended for more complex workflows where you want to use the OpenMS TOPP tools for instance with the **TOPP Workflow Framework**.
2. `Dockerfile_simple`: This Dockerfile builds only the Python packages. Recommended for simple apps using pyOpenMS only.

1. **Install Docker**

   Install Docker from the [official Docker installation guide](https://docs.docker.com/engine/install/)  
   
   <details>
   <summary>Click to expand</summary>
   
   ```bash
   # Remove older Docker versions (if any)
   for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt-get remove -y $pkg; done
   ```
   
   </details>

2. **Test Docker**
   
   Verify that Docker is working.
   ```bash
   docker run hello-world
   ```
   When running this command, you should see a hello world message from Docker.
   
3. **Clone the repository**
   ```bash
   git clone git@github.com:OpenMS/OpenDIAKiosk.git
   cd streamlit-template
   ```
   
4. **Specify GitHub token (to download Windows executables).**
   
   Create a temporary `.env` file with your Github token.
   
   It should contain only one line:
   `GITHUB_TOKEN=<your-github-token>`

   ℹ️ **Note:** This step is not strictly required, but skipping it will remove the option to download executables from the WebApp.
   
3. **Build & Launch the App**

   To build and start the containers.
   From the project root directory:
   
   ```bash
   docker-compose up -d --build
   ```
     At the end, you should see this:
      ```
      [+] Running 2/2
       ✔ openms-streamlit-template            Built      
       ✔ Container openms-streamlit-template  Started  
      ```
      
      To make sure server started successfully, run `docker compose ps`. You should see `Up` status:
      ```
      CONTAINER ID   IMAGE                       COMMAND                  CREATED         STATUS                 PORTS                                           NAMES
      4abe0603e521   openms_streamlit_template   "/app/entrypoint.sh …"   7 minutes ago   Up 7 minutes           0.0.0.0:8501->8501/tcp, :::8501->8501/tcp       openms-streamlit-template
      ```
   
      To map the port to default streamlit port `8501` and launch.
      
      ```
      docker run -p 8505:8501 openms_streamlit_template
      ```

   ### Mount a local data directory

   To make a directory of MS files on the host available to the running app
   without uploading or copying them, bind-mount it into the container at
   the path configured by `local_data_dir` in `settings.json` (the Docker
   image defaults this to `/mounted-data`):

   ```
   docker run -p 8501:8501 \
     -v /path/on/host:/mounted-data:ro \
     openms_streamlit_template
   ```

   The upload widget auto-detects the mount: when the directory exists at
   runtime it shows an in-app tree browser; selected files are referenced
   in place via `external_files.txt` (no copy into the workspace volume),
   so the mount can safely be read-only. Omitting `-v` hides the browser
   and falls back to the standard upload UI. To use a different container
   path, change `local_data_dir` in `settings.json` before building.

## Documentation

Documentation for **users** and **developers** is included as pages in [this template app](https://abi-services.cs.uni-tuebingen.de/streamlit-template/), indicated by the 📖 icon.

## Citation

Please cite:
Müller, T. D., Siraj, A., et al. OpenMS WebApps: Building User-Friendly Solutions for MS Analysis. Journal of Proteome Research (2025). [https://doi.org/10.1021/acs.jproteome.4c00872](https://doi.org/10.1021/acs.jproteome.4c00872)

## References

- Pfeuffer, J., Bielow, C., Wein, S. et al. OpenMS 3 enables reproducible analysis of large-scale mass spectrometry data. Nat Methods 21, 365–367 (2024). [https://doi.org/10.1038/s41592-024-02197-7](https://doi.org/10.1038/s41592-024-02197-7)

- Röst HL, Schmitt U, Aebersold R, Malmström L. pyOpenMS: a Python-based interface to the OpenMS mass-spectrometry algorithm library. Proteomics. 2014 Jan;14(1):74-7. [https://doi.org/10.1002/pmic.201300246](https://doi.org/10.1002/pmic.201300246). PMID: [24420968](https://pubmed.ncbi.nlm.nih.gov/24420968/).


