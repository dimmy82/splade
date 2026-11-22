# /bin/bash

curl -v -XPOST -H "content-type: application/json" "localhost:8000/encode?filterStopWords=false" -d@request2.json | jq .