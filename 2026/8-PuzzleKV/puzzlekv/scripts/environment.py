import importlib.metadata,json,platform,torch

def version(name):
    try: return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: return None
print(json.dumps(dict(platform=platform.platform(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,triton=version("triton"),transformers=version("transformers"),bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else None),indent=2))
