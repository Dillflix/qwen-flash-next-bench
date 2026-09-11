#!/usr/bin/env bash
# Existing-service migration only. Never sources an environment file as shell code.
set -Eeuo pipefail
umask 077
SERVICE=qwen-flash-next.service
DROPIN=/etc/systemd/system/qwen-flash-next.service.d/90-strix-runtime.conf
LAUNCHER=/usr/local/libexec/qwen-strix-production.py
SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if (( EUID != 0 )); then
    echo "Run with sudo bash deployment/install-strix-production.sh" >&2
    exit 64
fi

rollback() {
    local backup=$1
    case "$backup" in /var/backups/qwen-strix.*) ;; *) echo 'Invalid backup path' >&2; return 1 ;; esac
    [[ ${backup#/var/backups/} != */* ]] || return 1
    [[ -d "$backup" && ! -L "$backup" && -f "$backup/backup-complete" ]] || return 1
    [[ $(stat -c %u "$backup") == 0 ]] || return 1
    [[ -f "$backup/was-active" && -f "$backup/was-enabled" ]] || return 1
    case "$(< "$backup/was-active")" in 0|1) ;; *) return 1 ;; esac
    case "$(< "$backup/was-enabled")" in enabled|disabled) ;; *) return 1 ;; esac
    systemctl stop "$SERVICE" || return 1
    if [[ -f "$backup/previous-dropin" ]]; then
        install -m0644 "$backup/previous-dropin" "$DROPIN" || return 1
    else
        rm -f -- "$DROPIN" || return 1
    fi
    if [[ -f "$backup/previous-launcher" ]]; then
        install -m0755 "$backup/previous-launcher" "$LAUNCHER" || return 1
    else
        rm -f -- "$LAUNCHER" || return 1
    fi
    systemctl daemon-reload || return 1
    if [[ $(< "$backup/was-enabled") == disabled ]]; then
        systemctl disable "$SERVICE" || return 1
    else
        systemctl enable "$SERVICE" || return 1
    fi
    if [[ $(< "$backup/was-active") == 1 ]]; then
        systemctl start "$SERVICE" || return 1
    fi
    echo 'Previous runtime files and prior active/inactive state restored.'
}

if [[ ${1:-} == --rollback && $# == 2 ]]; then
    rollback "$2"
    exit
elif (( $# != 0 )); then
    echo 'Usage: install-strix-production.sh [--rollback /var/backups/qwen-strix.XXXXXX]' >&2
    exit 64
fi

[[ $(systemctl show "$SERVICE" -p LoadState --value) == loaded ]] || {
    echo 'Existing qwen-flash-next.service is required' >&2; exit 66;
}
[[ $(systemctl show "$SERVICE" -p User --value) == jdillman ]] || {
    echo 'Expected existing service User=jdillman; inspect the unit before migration' >&2; exit 66;
}
enabled=$(systemctl is-enabled "$SERVICE" 2>/dev/null || true)
[[ "$enabled" == enabled || "$enabled" == disabled ]] || {
    echo "Unsupported service enablement state: $enabled; no changes made" >&2; exit 66;
}
[[ -r /etc/qwen-flash-next.env && -f "$SOURCE_DIR/run-strix-production.py" ]] || exit 66
[[ ! -L "$DROPIN" && ! -L "$LAUNCHER" ]] || {
    echo 'Refusing to overwrite a symlinked Strix deployment file' >&2; exit 66;
}
# Do not collide with a manually running trial, even if it uses another port.
main_pid=$(systemctl show "$SERVICE" -p MainPID --value)
while read -r pid; do
    if [[ -n "$pid" && "$pid" != "$main_pid" ]]; then
        echo "A separate Strix server is running (PID $pid); stop that trial first." >&2
        exit 66
    fi
done < <(pgrep -f '^/srv/llm/src/strix-llama-trial-5f851647/build-hip10-dual/bin/llama-server( |$)' || true)

install -d -m0755 /var/backups
backup=$(mktemp -d /var/backups/qwen-strix.XXXXXX)
[[ ! -e "$DROPIN" ]] || cp -p -- "$DROPIN" "$backup/previous-dropin"
[[ ! -e "$LAUNCHER" ]] || cp -p -- "$LAUNCHER" "$backup/previous-launcher"
systemctl cat "$SERVICE" > "$backup/previous-effective-unit.txt"
if systemctl is-active --quiet "$SERVICE"; then
    printf '1\n' > "$backup/was-active"
else
    printf '0\n' > "$backup/was-active"
fi
printf '%s\n' "$enabled" > "$backup/was-enabled"
touch "$backup/backup-complete"
echo "Rollback backup: $backup"
echo "Rollback command: sudo bash $SOURCE_DIR/install-strix-production.sh --rollback $backup"
complete=0
recover() {
    local status=$?
    if (( complete == 0 )); then
        echo 'Deployment did not complete; rolling back...' >&2
        if ! rollback "$backup"; then
            echo "ROLLBACK FAILED; inspect systemctl status $SERVICE and backup $backup" >&2
        fi
    fi
    exit "$status"
}
trap recover EXIT
systemctl stop "$SERVICE"
install -d -m0755 /usr/local/libexec /etc/systemd/system/qwen-flash-next.service.d
install -m0755 "$SOURCE_DIR/run-strix-production.py" "$LAUNCHER"
install -m0644 "$SOURCE_DIR/90-strix-runtime.conf" "$DROPIN"
systemctl daemon-reload
echo 'Starting Strix and waiting for the authenticated readiness check (see journalctl -fu qwen-flash-next.service)...'
systemctl restart "$SERVICE"
systemctl is-active --quiet "$SERVICE"
systemctl enable "$SERVICE"
systemctl status "$SERVICE" --no-pager --lines=0
complete=1
echo 'Strix production deployment succeeded; enabled at boot.'
echo 'Existing authentication, bind address, port and old-runtime configuration were retained.'
