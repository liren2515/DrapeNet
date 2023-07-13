#!/bin/bash/

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

echo -e "\nEnter your name and email to configure Git"
read -p "Your first name:" firstname
read -p "You last name:" lastname
read -p "Email:" email
firstname=$(urle $firstname)
lastname=$(urle $lastname)
email=$(urle $email)

git config --global user.name "$firstname $lastname"
git config --global user.email $email 

