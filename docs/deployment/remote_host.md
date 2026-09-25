# Remote Host Setup

The repository keeps host addresses and credentials outside version control.
Create `~/.config/agentic_hardware/remote.env`:

```bash
AGENTIC_HW_SSH_KEY=/absolute/path/to/private_key
AGENTIC_HW_REMOTE_USER=your-login
AGENTIC_HW_REMOTE_HOSTS="203.0.113.10"
AGENTIC_HW_REMOTE_DIR=/home/your-login/agentic_hardware
```

Upload, connect, check readiness, and download artifacts:

```bash
infra/remote/upload_workspace.sh 0
infra/remote/connect.sh 0
infra/remote/check_host_ready.sh 0
infra/remote/download_artifacts.sh 0
```

On the remote host, select the matching capability profile before running an
experiment. For an A10G-class device, use `nvidia_a10g_24gb`.
