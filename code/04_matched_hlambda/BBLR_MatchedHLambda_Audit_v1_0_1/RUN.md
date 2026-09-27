# Linux run sequence

Project root expected by the project workflow:

`/media/simon/E490B01790AFEE6C/BBLR2`

After manually placing the ZIP and SHA file there, verify and unzip. Then enter the standalone
folder, use the existing project virtual environment if appropriate (or create `.venv` locally),
install requirements, run tests, run the scientific preflight, then production.

Recommended production settings: 140 workers, 25 frozen replications per shard, resume enabled.
Raw shards remain on the workstation. After completion, run analysis and create the compact handback.
