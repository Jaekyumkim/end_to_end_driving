How to use Udacity Simulator
===============================
 
- Data Recording

  Default Linux desktop Universal.x86_64을 실행시키고 Training Mode로 들어가서 우상단의 RECORD를 누르고 폴더를 지정한 후 화살표로 주행하다가 정지를 누르면 그동안 주행한 데이터가 저장된다. 주행데이터는 **left, center, right camera이미지**와 **steering angle, throttle, speed**가 저장된다
  
- Training
```python
python model.py -d /driving_log_directory
```
 이렇게 하면 좋은 트레이닝 결과가 나올 때 마다 model-006.h5같은 h5파일이 생성된다

 - Test
```python
python drive.py model-006.h5
```
이후 Udacity_Simulator 폴더에 있는 Default Linux desktop Universal.x86_64를 실행시키고 AUTONOMOUS MODE로 들어가면 자동으로 통신해서 테스트할 수 있다

 - File
 
   NAS/1/end_driving/udacity_simulator에 업로드 해놓음
  
https://github.com/naokishibuya/car-behavioral-cloning/blob/master/README.md
