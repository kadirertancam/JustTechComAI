FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Sistem paketleri
RUN apt-get update && \
    apt-get install -y git python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Python kısayolu
RUN ln -s /usr/bin/python3 /usr/bin/python

# Gerekli Python kütüphaneleri
RUN pip install --upgrade pip && \
    pip install transformers accelerate sentencepiece gradio bitsandbytes

# Çalışma dizini
WORKDIR /app

# Model dosyalarını önceden indirmek (isteğe bağlı, cache amaçlı)
# RUN python - <<EOF
# from transformers import AutoTokenizer, AutoModelForCausalLM
# model_name = "gpt-oss/gpt-oss-20b"
# AutoTokenizer.from_pretrained(model_name)
# AutoModelForCausalLM.from_pretrained(model_name)
# EOF

# Ana uygulama dosyasını kopyala
COPY app.py /app/app.py

# Container başlatıldığında çalışacak komut
CMD ["python", "app.py"]
