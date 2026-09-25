# Remote Host Helpers

These helpers upload the workspace, connect to a remote host, download
artifacts, and check host readiness. They contain no machine identity or
credentials.

Create `~/.config/agentic_hardware/remote.env` outside this repository:

```bash
AGENTIC_HW_SSH_KEY=/absolute/path/to/private_key
AGENTIC_HW_REMOTE_USER=your-login
AGENTIC_HW_REMOTE_HOSTS="203.0.113.10"
AGENTIC_HW_REMOTE_DIR=/home/your-login/agentic_hardware
```

Then use `infra/remote/upload_workspace.sh`, `connect.sh`,
`download_artifacts.sh`, or `check_host_ready.sh`.
