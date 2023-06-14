#!/bin/bash

USERNAME_FILE="$HOME/.hpi_username"
SNXRC_FILE="$HOME/.snxrc"
MOUNT_DIR="/mnt/hpi/fg"
DNS_SERVER="10.10.10.12"

if [ ! -f "$USERNAME_FILE" ]; then
    read -p 'HPI username: ' username

    rm "$SNXRC_FILE"
    echo "server vpn.hpi.de" >> "$SNXRC_FILE"
    echo "username ${username}" >> "$SNXRC_FILE"
    echo -e "Created VPN config file $SNXRC_FILE"

    echo "${username}" > "${USERNAME_FILE}"
    echo -e "Saved HPI username to ${USERNAME_FILE}"
else
    username=$(cat "${USERNAME_FILE}")
    echo -e "HPI username found in ${USERNAME_FILE}"
fi

echo -e "\nConnecting to HPI VPN..."
snx

echo -e "\n\n"

if grep -q "$DNS_SERVER" /etc/resolv.conf; then
    echo -e "HPI DNS ($DNS_SERVER) is already added to /etc/resolv.conf"
else
    sudo echo "nameserver $DNS_SERVER" >> /etc/resolv.conf
    echo -e "Added HPI DNS ($DNS_SERVER) to /etc/resolv.conf"
fi

if mountpoint -q "$MOUNT_DIR"; then
    echo -e "\nFG share is already mounted at $MOUNT_DIR"
else
    echo -e "\nMounting FG share..."
    sudo mount -t smb3 -o username="${username}",domain=hpi //store-01.hpi.uni-potsdam.de/fg "$MOUNT_DIR"
    echo -e "\nFG share mounted successfully!"
fi