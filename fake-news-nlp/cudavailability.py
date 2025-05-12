import torch
print(torch.__version__)          # Powinno pokazywać wersję >1.10
print(torch.cuda.is_available())  # Powinno zwrócić True
print(torch.version.cuda)         # Powinno zgadzać się z wersją z `nvidia-smi` (12.x)