#!/bin/bash/

urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# SMPL model
echo -e "\nYou need to register at https://smpl.is.tue.mpg.de"
read -p "Username (SMPL):" username
read -p "Password (SMPL):" password
username=$(urle $username)
password=$(urle $password)

wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip' -O 'SMPL_python_v.1.0.0.zip' --no-check-certificate --continue
unzip SMPL_python_v.1.0.0.zip \
	&& mv smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl smpl_pytorch/ \
	&& rm -r smpl SMPL_python_v.1.0.0.zip __MACOSX

