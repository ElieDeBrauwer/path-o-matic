#!/bin/sh
curl --user barco:b4rc0,BCD -k -i -X POST https://35.197.102.39/app/classify -H "Content-Type: text/xml" --data-binary "@./7.png"
