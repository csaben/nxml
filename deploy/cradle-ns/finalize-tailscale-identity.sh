#!/usr/bin/env bash
set -euo pipefail

expected_user=arelius
expected_backend=http://127.0.0.1:8090
expected_host=cradle-ns.tailb1b51d.ts.net

if [[ ${EUID} -ne 0 ]]; then
    echo "run this finalizer once with sudo" >&2
    exit 1
fi
if [[ $(hostname) != cradle ]]; then
    echo "refusing unexpected host: $(hostname)" >&2
    exit 1
fi
id "$expected_user" >/dev/null
test -f /etc/polkit-1/rules.d/49-nxml-edge.rules
test -f /etc/systemd/system/nxml-bt.service
grep -Fq 'unit === "nxml-bt.service"' /etc/polkit-1/rules.d/49-nxml-edge.rules
grep -Fq 'subject.user !== "arelius"' /etc/polkit-1/rules.d/49-nxml-edge.rules
grep -Fq -- '--reconnect-address 58:2F:40:23:3C:CA' /etc/systemd/system/nxml-bt.service

# polkit started before /etc/polkit-1/rules.d existed on this host. Reload it
# once so its watcher and the narrow nxml-bt authorization are authoritative.
systemctl restart polkit.service
systemctl is-active --quiet polkit.service
runuser -u "$expected_user" -- \
    systemctl --no-ask-password start nxml-bt.service
systemctl is-active --quiet nxml-bt.service

# Delegate only Tailscale CLI configuration to the logged-in operator, then
# install a private Tailnet Serve proxy. This never enables Funnel.
tailscale set --operator="$expected_user"
tailscale serve --bg --yes "$expected_backend"
tailscale serve status | grep -Fq "https://${expected_host}"

echo "NXML identity proxy finalized: https://${expected_host}"
