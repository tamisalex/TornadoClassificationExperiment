printf "Tornado Pipeline starting...\n"
printf "Collecting Storm Data ...\n"

python StormEventDataCollector.py

printf "Collecting Radar Data...\n"

python RadarDataCollector.py

printf "Processing Radar Data...\n"

python RadarDataProcessor.py

printf "Modeling...\n"

python TornadoModelAndMetrics.py