# Cradle deployment runbook

These are templates only. Do not place a real token in Git.

Administrator provisioning:

```sh
sudo useradd --system --home-dir /var/lib/nxml-control --shell /usr/sbin/nologin nxml-control
sudo install -d -o nxml-control -g nxml-control -m 0750 /var/lib/nxml-control /var/lib/nxml-control/checkpoints /var/log/nxml-control
sudo install -d -o root -g nxml-control -m 0750 /etc/nxml-control
sudo install -o nxml-control -g nxml-control -m 0600 /dev/null /etc/nxml-control/token
openssl rand -hex 32 | sudo tee /etc/nxml-control/token >/dev/null
sudo install -o root -g nxml-control -m 0640 apps/nxml-control/deploy/nxml-control.env.example /etc/nxml-control/nxml-control.env
sudo install -o root -g root -m 0644 apps/nxml-control/deploy/nxml-control.service /etc/systemd/system/nxml-control.service
sudo systemctl daemon-reload
sudo systemctl enable --now nxml-control.service
```

The checkout/venv at `/opt/nxml` must already be installed and readable by the service user. If its location differs, edit `ExecStart` before installation. Confirm cradle owns Tailnet address `100.80.98.4`; do not start the service if it does not.

Readiness and authenticated checks:

```sh
curl --fail http://100.80.98.4:8787/healthz
sudo -u nxml-control sh -c 'TOKEN=$(cat /etc/nxml-control/token); curl --fail -H "Authorization: Bearer $TOKEN" http://100.80.98.4:8787/v1/deployments'
sudo systemctl status nxml-control.service
sudo journalctl -u nxml-control.service --since today
```

Provision the same token to nxml-edge through its secret manager or a local 0600 file; its HTTP client must send `Authorization: Bearer <token>`. Never copy the token into an environment example, command history, unit file, repository, issue, or log. Rotate by atomically replacing `/etc/nxml-control/token` with another 0600 file and restarting both client and service in a coordinated maintenance window.

The ZMQ+frames 30fps inference data plane remains a separate process/socket. This HTTP service is only the authenticated control plane.
