import jsonschema

MODEL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "version": {"type": "string"},
        "author": {"type": "string"},
        "description": {"type": "string"},
        "supported_device": {"type": "string"},
        "input_schema": {"type": "string"},
        "output_schema": {"type": "string"},
        "hash": {"type": "string"},
        "model_type": {"type": "string"},
        "framework": {"type": "string"},
        "signature": {"type": "string"},
    },
    "required": [
        "name",
        "version",
        "author",
        "description",
        "supported_device",
        "input_schema",
        "output_schema",
        "hash",
        "model_type",
        "framework",
    ],
}


def validate_model_metadata(metadata):
    jsonschema.validate(instance=metadata, schema=MODEL_SCHEMA)
