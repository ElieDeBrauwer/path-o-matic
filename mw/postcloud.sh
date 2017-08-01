#!/bin/sh

if [ $# -lt 4 ]
then
	echo "Usage: sh postcloud.sh <user> <password> <serverip> <pngfile>"
	exit
fi


USER=$1
PASSWORD=$2
SERVERIP=$3
PNGFILE=$4

curl --user ${USER}:${PASSWORD} -k -i -X POST https://${SERVERIP}/app/classify -H "Content-Type: text/xml" --data-binary "@${PNGFILE}"
