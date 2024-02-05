## 설치 장비 확인
### 카이로스
- CPU : Intel 13세대 i7-1355U
- GPU : NVIDIA Geforce RTX 2050
### 집
- GPU : Nvitia Geforce RTX3070ti
## 설치 메뉴얼
### GPU에 맞는 CUDA 확인
- [GPU Compute Capability]
  (https://developer.nvidia.com/cuda-gpus)![[Pasted image 20240204112556.png|]]
- [Compute Capability for Cuda Version]
  (https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications)![[Pasted image 20240204113210.png]]
### Tensorflow에 맞는 Cuda 확인
- [tensorflow with python and Cuda(Cudnn)](https://www.tensorflow.org/install/source?hl=ko)
### 설치
1. 쿠다를 설치한다 (.exe)
2. Cudnn을 풀어준다.
##### 윈도우 (GPU)
![[Pasted image 20240204112723.png]]
##### 리눅스 (GPU)
![[Pasted image 20240204112758.png]]

## 리눅스 최신버전 설치 간단 메뉴얼
### 간단 설치 메뉴얼 (윈도우11 이상)
1. WSL을 설치한다 (기본옵션은 Ubuntu, python 3.10.12 설치 됨)
   `wsl --install -d ubuntu-20.04` / `wsl --install`
2. WSL2 - CUDA, Cudnn 설치
	- GPU에 맞는 CUDA버전 확인 
		- [Compute Capablity](https://developer.nvidia.com/cuda-gpus) 3070ti : 8.6, 4050 : 8.9 (파이썬 3.9~3.11)
		- CUDA Ver : 12.0 ~ 12.4 (Tensorflow 2.15.0 - Cuda 12.2)
	- Cuda 설치 (리눅스 - wsl)![[Cuda_Linux_12_2.txt]]
	- Cudnn![[cudnn.txt]]
3. 텐서플로우 설치
	1. `sudo apt update`
	2. `sudo apt install python3-pip`
	3. `pip install tensorflow`
	4. 확인코드
```ad-todo
~~~python
import tensorflow as tf # 

# TensorFlow 버전 확인
print("TensorFlow version:", tf.__version__)

# 사용 가능한 GPU 목록을 출력
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# TensorFlow가 GPU를 사용하는지 여부 확인
if gpus:
    for gpu in gpus:
        print("GPU Name:", gpu.name, "GPU Type:", gpu.device_type)
else:
    print("TensorFlow is not using GPU.")
```
![[Pasted image 20240203192221.png]]
## TensorRT 설치
### 사용에 영향을 미칠 수 있는 부분:

1. **TensorRT 경고**:
    - 경고 메시지: "TF-TRT Warning: Could not find TensorRT"
    - 영향: TensorRT는 NVIDIA의 딥 러닝 추론 엔진으로, 모델을 최적화하여 GPU에서의 추론 성능을 향상시킵니다. 이 경고는 TensorRT가 설치되어 있지 않거나 시스템에서 찾을 수 없다는 것을 의미하며, TensorFlow의 TensorRT 통합 기능을 사용할 수 없음을 나타냅니다. 이로 인해 GPU에서 실행되는 딥 러닝 모델의 추론 성능이 최적화되지 않을 수 있습니다.
    - 해결 방법: TensorRT를 설치하고, 환경 변수를 올바르게 설정하여 TensorFlow가 TensorRT를 찾을 수 있도록 합니다. NVIDIA 공식 홈페이지에서 호환되는 TensorRT 버전을 다운로드하고 설치 지침을 따르세요. 설치 후, `LD_LIBRARY_PATH` 환경 변수에 TensorRT 라이브러리의 경로를 추가합니다.

   TensorRT 설치는 [다운로드](https://developer.nvidia.com/tensorrt-download)한 패키지의 형태(예: tar 파일, Debian 패키지)에 따라 다릅니다. Linux 시스템에서의 일반적인 설치 방법은 다음과 같습니다:

### 버전호한 
##### 윈도우 (Cuda 11.2, Cudnn8.1 - tensorrt 7.2.3)
- https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/support-matrix/index.html
- https://developer.nvidia.com/tensorrt-download
  ![[Pasted image 20240204121029.png]]
### 주의 사항
- **CUDA 버전 호환성**: 설치하려는 TensorRT 버전이 정확히 CUDA 12.2와 호환되지 않을 수 있습니다. 이런 경우, CUDA 12.0용 TensorRT를 설치하고, 실행 시 어떤 문제가 발생하는지 확인해야 합니다. 대부분의 경우, 소프트웨어가 호환 모드에서 정상 작동할 수 있습니다.
- **공식 문서 확인**: 설치하기 전에 항상 NVIDIA의 공식 문서를 확인하여, 선택한 TensorRT 버전이 시스템 구성과 호환되는지 확인해야 합니다. CUDA와 TensorRT의 호환성은 NVIDIA에 의해 정기적으로 업데이트되며, 최신 정보는 공식 문서에서 제공됩니다.
- WSL
  ![[Pasted image 20240205090703.png]]

## 참고자료
- [RTX3070 GPU 사용하기 (CUDA 11.2, cuDNN 설치) for Windows](https://foreverhappiness.tistory.com/123)
- https://webnautes.tistory.com/1875 
- **[딥러닝 환경 세팅 윈도우에서 Tensorflow GPU 환경 구성하기 (WSL2, CUDA, Anaconda)](https://21june.tistory.com/2)**
- https://webnautes.tistory.com/1848	
- [WSL2와 Windows에서 파일 접근하기](https://coding-nyan.tistory.com/155)


## 나중에 
- torch for CUDA 11.2
	- [PyTorch CUDA 11.2 + RTX3090에 맞는 torch version 세팅하기](https://daeun-computer-uneasy.tistory.com/60#)
- another ver.
	- https://pytorch.org/get-started/locally/
- https://codequeen.tistory.com/
-  [아나콘다 pip 전체 삭제](https://seong6496.tistory.com/91)