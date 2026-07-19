#!/usr/bin/env bash
set -euo pipefail

expected_user=arelius
expected_backend=http://127.0.0.1:8090
expected_host=cradle-ns.tailb1b51d.ts.net
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
rule_source=${script_dir}/polkit/49-nxml-edge.rules
rule_etc=/etc/polkit-1/rules.d/49-nxml-edge.rules
rule_vendor=/usr/share/polkit-1/rules.d/49-nxml-edge.rules

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

# Establish the private identity proxy before diagnostic authorization proofs
# so the UI cannot remain stranded behind loopback. This never enables Funnel.
tailscale set --operator="$expected_user"
tailscale serve --bg --yes "$expected_backend"
tailscale serve status | grep -Fq "https://${expected_host}"

# Use the real unprivileged edge process as the PolicyKit subject. pkcheck is
# run by root so it may supply the exact systemd unit/verb details, but it does
# not call systemd or change any service state.
edge_pid=$(pgrep -u "$(id -u "$expected_user")" -x nxml-edge | head -n 1)
test -n "$edge_pid"
edge_uid=$(stat -c %u "/proc/${edge_pid}")
edge_start=$(awk '{print $22}' "/proc/${edge_pid}/stat")
test "$edge_uid" = "$(id -u "$expected_user")"
edge_subject=${edge_pid},${edge_start},${edge_uid}
pkcheck --action-id org.freedesktop.systemd1.manage-units \
    --process "$edge_subject" \
    --detail unit nxml-bt.service \
    --detail verb start
if pkcheck --action-id org.freedesktop.systemd1.manage-units \
    --process "$edge_subject" \
    --detail unit ssh.service \
    --detail verb start; then
    echo "narrow PolicyKit proof failed: ssh.service was authorized" >&2
    exit 1
fi

echo "NXML identity proxy finalized: https://${expected_host}"
