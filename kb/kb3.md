Several resources and repositories provide practical software deployment examples for **Neural Processing Units (NPUs)**, ranging from local transcription services to large language model (LLM) implementations.

### Speech-to-Text (STT) and Audio Deployment
*   **whisper-npu-server**: This project implements a local transcription service running in a rootless Podman container, utilizing the **Intel NPU** to transcribe speech using the **OpenAI Whisper** model. 
    *   URL: [https://github.com/ellenhp/whisper-npu-server](https://github.com/ellenhp/whisper-npu-server)
*   **whisper-transcription-wayland**: A wrapper program that enables a global hotkey to record voice, transcribe it on the NPU, and type the result into a focused application.
    *   URL: [https://github.com/ellenhp/whisper-transcription-wayland/](https://github.com/ellenhp/whisper-transcription-wayland/)
*   **OpenVINO Whisper ASR Notebook**: An interactive Jupyter notebook detailing how to perform automatic speech recognition using **Whisper** and **OpenVINO GenAI**.
    *   URL: [https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-asr-genai](https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/whisper-asr-genai)

### Large Language Models (LLMs) and Generative AI
*   **tinyllama-on-intel-npu**: A deployment example of the **TinyLlama 1.1B Chat** model running locally on **Intel Core Ultra** hardware using **OpenVINO GenAI** with **INT4/FP16** quantization.
    *   URL: [https://github.com/balaragavan2007/tinyllama-on-intel-npu](https://github.com/balaragavan2007/tinyllama-on-intel-npu)
*   **OpenVINO Test Drive**: A beta console that demonstrates on-device GenAI scenarios, including chatbots and live captioning, optimized for Intel hardware.
    *   URL: [https://github.com/openvinotoolkit/openvino_testdrive](https://github.com/openvinotoolkit/openvino_testdrive)

### Multi-Platform and Vendor-Specific Model Zoos
*   **Rebellions Model Zoo**: Provides specific scripts (`compile.py` and `inference.py`) for deploying models such as **Llama-3**, **Stable Diffusion**, and **YOLO11** on the **Rebellions RBLN-CA12 (ATOM)** NPU server.
    *   URL: [https://github.com/rebellions-sw/rbln-model-zoo](https://github.com/rebellions-sw/rbln-model-zoo)
*   **Qualcomm AI Hub**: A resource for AI models validated and optimized specifically for **Qualcomm NPUs** found in Copilot+ PCs.
    *   URL: Access via [Qualcomm Developer Network](https://www.qualcomm.com/developer/artificial-intelligence) (outside the sources, but referenced in context).
*   **ONNX Model Zoo**: A curated collection of pre-trained models in the **ONNX** format recommended for use with NPUs across various devices.
    *   URL: [https://github.com/onnx/models](https://github.com/onnx/models) (outside the sources, but referenced in context).

### Foundational Tutorials and Documentation
*   **OpenVINO Hello NPU Notebook**: A basic introduction to configuring and running inference on an Intel NPU device.
    *   URL: [https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/hello-npu/hello-npu.ipynb](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/hello-npu/hello-npu.ipynb)
*   **Intel AI PC Notebooks**: A repository containing code samples for tasks like **AI Upscaling** (BSRGAN) specifically optimized for the **Intel AI Boost** NPU.
    *   URL: [https://github.com/intel/ai-pc-notebooks](https://github.com/intel/ai-pc-notebooks) (outside the sources, but referenced in context).