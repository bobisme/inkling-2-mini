maya_dir := env("MAYA_DIR", home_directory() / "models/inkling/src/maya")
model_gguf := env("MODEL_GGUF", home_directory() / "models/inkling/Inkling-2-Mini-abliterated-orca-UD-Q4_K_XL-maya.gguf")

# Serve the spliced abliterated model with Maya on the local GPU (foreground; Ctrl+C stops it).
# config: ik2-abliterated, ik2-abliterated-nodraft or ik2-local (stock weights). pi model: maya/inkling
# MAYA_DIR is the Maya checkout; MODEL_GGUF is the output of tools/splice_abliterated.py.
maya config="ik2-abliterated" port="18000":
    sed 's#@MODEL_GGUF@#{{model_gguf}}#' maya/{{config}}.json > "{{maya_dir}}/{{config}}.json"
    cd "{{maya_dir}}" && exec .venv/bin/python serve/server.py --engine maya --config {{config}}.json --host 127.0.0.1 --port {{port}}
