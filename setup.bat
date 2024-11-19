python -m venv venv
call venv\\Scripts\\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade diffusers==0.30.3 transformers==4.45.1
pip install accelerate ftfy numpy Pillow safetensors opencv-python
python -m pip install notebook
call venv\\Scripts\\deactivate.bat
pause