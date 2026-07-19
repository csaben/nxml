#!/usr/bin/env bash
set -euo pipefail

expected_user=arelius
expected_backend=http://127.0.0.1:8090
expected_host=cradle-ns.tailb1b51d.ts.net
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rule_source=${script_dir}/polkit/49-nxml-edge.rules
rule_etc=/etc/polkit-1/rules.d/49-nxml-edge.rules
rule_vendor=/usr/share/polkit-1/rules.d/49-nxml-edge.rules
denial_log=$(mktemp /tmp/nxml-polkit-denial.XXXXXX)
trap 'rm -f "$denial_log"' EXIT

if [[ ${EUID} -ne 0 ]]; then
    echo "run this finalizer once with sudo" >&2
    exit 1
fi
if [[ $(hostname) != cradle ]]; then
    echo "refusing unexpected host: $(hostname)" >&2
    exit 1
fi
id "$expected_user" >/dev/null
test -f /etc/systemd/system/nxml-bt.service
test "$(id -u "$expected_user")" = 1000
test -f "$rule_source"
grep -Fq 'unit === "nxml-bt.service"' "$rule_source"
grep -Fq 'subject.user !== "arelius"' "$rule_source"
grep -Fq -- '--reconnect-address 58:2F:40:23:3C:CA' /etc/systemd/system/nxml-bt.service

# This Ubuntu polkit 0.105 host loads packaged JavaScript rules from the
# existing /usr/share path; /etc/polkit-1/rules.d was absent at daemon start.
# Keep the admin copy and install the identical rule into the active path.
install -d -m 0755 -o root -g root /etc/polkit-1/rules.d
install -m 0644 -o root -g root "$rule_source" "$rule_etc"
install -m 0644 -o root -g root "$rule_source" "$rule_vendor"
systemctl restart polkit.service
systemctl is-active --quiet polkit.service

# Starting an already-active unit is non-mutating but still exercises the
# exact systemd D-Bus/PolicyKit authorization path as arelius.
runuser -u "$expected_user" -- \
    systemctl --no-ask-password start nxml-bt.service
systemctl is-active --quiet nxml-bt.service

# Prove that the same caller cannot manage a different unit. Require ssh to be
# active first so even an unexpected authorization could not start it.
systemctl is-active --quiet ssh.service
if runuser -u "$expected_user" -- \
    systemctl --no-ask-password start ssh.service 2>"$denial_log"; then
    echo "narrow PolicyKit proof failed: ssh.service was authorized" >&2
    exit 1
fi
grep -Fq 'Interactive authentication required' "$denial_log"

# Delegate only Tailscale CLI configuration to the logged-in operator, then
# install a private Tailnet Serve proxy. This never enables Funnel.
tailscale set --operator="$expected_user"
tailscale serve --bg --yes "$expected_backend"
tailscale serve status | grep -Fq "https://${expected_host}"

echo "NXML identity proxy finalized: https://${expected_host}"
